#!/bin/bash

trtexec --onnx=bmshj2018_factorized_g_a_768x512.onnx --fp16 --saveEngine=bmshj2018_factorized_g_a_768x512.trt --workspace=4096
trtexec --onnx=bmshj2018_factorized_g_s_768x512.onnx --fp16 --saveEngine=bmshj2018_factorized_g_s_768x512.trt --workspace=4096
trtexec --onnx=bmshj2018_factorized_g_a_1280x720.onnx --fp16 --saveEngine=bmshj2018_factorized_g_a_1280x720.trt --workspace=4096
trtexec --onnx=bmshj2018_factorized_g_s_1280x720.onnx --fp16 --saveEngine=bmshj2018_factorized_g_s_1280x720.trt --workspace=4096
