#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:50:48 2021

@author: arslan
"""
import taso
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to input ONNX file", required=True)

args = parser.parse_args()


graph = taso.load_onnx(args.file)
new_graph = taso.optimize(graph, alpha = 1.0, budget = 100, print_subst=True)
onnx_model = taso.export_onnx(new_graph)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "optimised_{}".format(args.file))
