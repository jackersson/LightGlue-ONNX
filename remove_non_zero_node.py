import onnx
import fire

def remove_non_zero_node(model_path: str, output_path: str):
    model = onnx.load(model_path)
    nodes = model.graph.node
    
    # remove NonZero node and connect previous node to next node
    for i, node in enumerate(nodes):
        if node.op_type == 'NonZero':
            prev_node = nodes[i-1]
            next_node = nodes[i+1]
            print(node)
            next_node.input[0] = prev_node.output[0]
            nodes.pop(i)
            break
    
    onnx.save(model, output_path)
    print(f'Model saved to: {output_path}')
    
if __name__ == '__main__':
    fire.Fire(remove_non_zero_node)