# TAKEN FROM https://github.com/NVIDIA/TensorRT/blob/release/10.4/demo/Diffusion/utils_modelopt.py
#

import re
import numpy as np
import onnx
import onnx_graphsurgeon as gs


def get_parent_nodes(node):
    """
    Returns list of input producer nodes for the given node.
    """
    parents = []
    for tensor in node.inputs:
        # If the tensor is not a constant or graph input and has a producer,
        # the producer is a parent of node `node`
        if len(tensor.inputs) == 1:
            parents.append(tensor.inputs[0])
    return parents


def get_child_nodes(node):
    """
    Returns list of output consumer nodes for the given node.
    """
    children = []
    for tensor in node.outputs:
        for consumer in tensor.outputs:  # Traverse all consumer of the tensor
            children.append(consumer)
    return children


def has_path_type(node, graph, path_type, is_forward, wild_card_types, path_nodes):
    """
    Return pattern nodes for the given path_type.
    """
    if not path_type:
        # All types matched
        return True

    # Check if current non-wild node type does not match the expected path type
    node_type = node.op
    is_match = node_type == path_type[0]
    is_wild_match = node_type in wild_card_types
    if not is_match and not is_wild_match:
        return False

    if is_match:
        path_nodes.append(node)
        next_path_type = path_type[1:]
    else:
        next_path_type = path_type[:]

    if is_forward:
        next_level_nodes = get_child_nodes(node)
    else:
        next_level_nodes = get_parent_nodes(node)

    # Check if any child (forward path) or parent (backward path) can match the remaining path types
    for next_node in next_level_nodes:
        sub_path = []
        if has_path_type(next_node, graph, next_path_type, is_forward, wild_card_types, sub_path):
            path_nodes.extend(sub_path)
            return True

    # Path type matches if there is no remaining types to match
    return not next_path_type


def insert_cast(graph, input_tensor, attrs):
    """
    Create a cast layer using tensor as input.
    """
    output_tensor = gs.Variable(name=f"{input_tensor.name}/Cast_output", dtype=attrs["to"])
    next_node_list = input_tensor.outputs.copy()
    graph.layer(
        op="Cast",
        name=f"{input_tensor.name}/Cast",
        inputs=[input_tensor],
        outputs=[output_tensor],
        attrs=attrs,
    )

    # use cast output as input to next node
    for next_node in next_node_list:
        for idx, next_input in enumerate(next_node.inputs):
            if next_input.name == input_tensor.name:
                next_node.inputs[idx] = output_tensor


def convert_zp_fp8(onnx_graph):
    """
    Convert Q/DQ zero datatype from INT8 to FP8.
    """
    # Find all zero constant nodes
    qdq_zero_nodes = set()
    for node in onnx_graph.graph.node:
        if node.op_type == "QuantizeLinear":
            if len(node.input) > 2:
                qdq_zero_nodes.add(node.input[2])

    print(f"Found {len(qdq_zero_nodes)} QDQ pairs")

    # Convert zero point datatype from INT8 to FP8.
    for node in onnx_graph.graph.node:
        if node.output[0] in qdq_zero_nodes:
            node.attribute[0].t.data_type = onnx.TensorProto.FLOAT8E4M3FN

    return onnx_graph


def cast_resize_io(graph):
    """
    After all activations and weights are converted to fp16, we will
    add cast nodes to Resize nodes I/O because Resize need to be run in fp32.
    """
    nodes = graph.nodes
    up_block_resize_regex = r"\/up_blocks.[0-2]\/upsamplers.0\/Resize"
    up_block_resize_nodes = [_n for _n in nodes if re.match(up_block_resize_regex, _n.name)]

    print(f"Found {len(up_block_resize_nodes)} Resize nodes to fix")
    for resize_node in up_block_resize_nodes:
        for input_tensor in resize_node.inputs:
            if input_tensor.name:
                insert_cast(graph, input_tensor=input_tensor, attrs={"to": np.float32})
        for output_tensor in resize_node.outputs:
            if output_tensor.name:
                insert_cast(graph, input_tensor=output_tensor, attrs={"to": np.float16})


def cast_fp8_mha_io(graph):
    r"""
    Insert three cast ops.
    The first cast will be added before the input0 of MatMul to cast fp16 to fp32.
    The second cast will be added before the input1 of MatMul to cast fp16 to fp32.
    The third cast will be added after the output of MatMul to cast fp32 back to fp16.
        Q                  Q
        |                  |
        DQ                 DQ
        |                  |
        Cast               Cast
    (fp16 to fp32)    (fp16 to fp32)
        \                  /
          \              /
            \          /
              MatMul
                |
               Cast (fp32 to fp16)
                |
                Q
                |
                DQ
    The insertion of Cast ops in the FP8 MHA part actually forbids the MHAs to run
    with FP16 accumulation because TensorRT only has FP32 accumulation kernels for FP8 MHAs.
    """
    # Find FP8 MHA pattern.
    # Match FP8 MHA: Q -> DQ -> BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> Q -> DQ -> BMM2 -> Q -> DQ
    softmax_bmm1_chain_type = ["Softmax", "MatMul", "DequantizeLinear", "QuantizeLinear"]
    softmax_bmm2_chain_type = [
        "Softmax",
        "QuantizeLinear",
        "DequantizeLinear",
        "MatMul",
        "QuantizeLinear",
        "DequantizeLinear",
    ]
    wild_card_types = [
        "Div",
        "Mul",
        "ConstMul",
        "Add",
        "BiasAdd",
        "Reshape",
        "Transpose",
        "Flatten",
        "Cast",
    ]

    fp8_mha_partitions = []
    for node in graph.nodes:
        if node.op == "Softmax":
            fp8_mha_partition = []
            if has_path_type(
                node, graph, softmax_bmm1_chain_type, False, wild_card_types, fp8_mha_partition
            ) and has_path_type(node, graph, softmax_bmm2_chain_type, True, wild_card_types, fp8_mha_partition):
                if (
                    len(fp8_mha_partition) == 10
                    and fp8_mha_partition[1].op == "MatMul"
                    and fp8_mha_partition[7].op == "MatMul"
                ):
                    fp8_mha_partitions.append(fp8_mha_partition)

    print(f"Found {len(fp8_mha_partitions)} FP8 attentions")

    # Insert Cast nodes for BMM1 and BMM2.
    for fp8_mha_partition in fp8_mha_partitions:
        bmm1_node = fp8_mha_partition[1]
        insert_cast(graph, input_tensor=bmm1_node.inputs[0], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm1_node.inputs[1], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm1_node.outputs[0], attrs={"to": np.float16})

        bmm2_node = fp8_mha_partition[7]
        insert_cast(graph, input_tensor=bmm2_node.inputs[0], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm2_node.inputs[1], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm2_node.outputs[0], attrs={"to": np.float16})


def convert_fp16_io(graph):
    """
    Convert graph I/O to FP16.
    """
    for input_tensor in graph.inputs:
        input_tensor.dtype = onnx.TensorProto.FLOAT16
    for output_tensor in graph.outputs:
        output_tensor.dtype = onnx.TensorProto.FLOAT16
