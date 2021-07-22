import taso
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to input ONNX file", required=True)

args = parser.parse_args()

graph = taso.load_onnx(args.file)
new_graph = taso.optimize(graph, alpha = 1.0, budget = 100, print_subst=True)

print("Original Graph cost: "+str(graph.run_time()))
print("Optimised Graph cost: "+str(new_graph.run_time()))

onnx_model = taso.export_onnx(new_graph)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "{}.taso.onnx".format(args.file))
