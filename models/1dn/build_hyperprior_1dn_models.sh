#!/bin/bash

trtexec --onnx=bmshj2018_hyperprior_1dn_q2_compress_768x512.onnx --fp16 --saveEngine=bmshj2018_hyperprior_1dn_q2_compress_768x512.trt --workspace=4096
trtexec --onnx=bmshj2018_hyperprior_1dn_q2_h_s_768x512.onnx --fp16 --saveEngine=bmshj2018_hyperprior_1dn_q2_h_s_768x512.trt --workspace=4096
trtexec --onnx=bmshj2018_hyperprior_1dn_q2_g_s_768x512.onnx --fp16 --saveEngine=bmshj2018_hyperprior_1dn_q2_g_s_768x512.trt --workspace=4096
trtexec --onnx=bmshj2018_hyperprior_1dn_q2_compress_1280x720.onnx --fp16 --saveEngine=bmshj2018_hyperprior_1dn_q2_compress_1280x720.trt --workspace=4096
trtexec --onnx=bmshj2018_hyperprior_1dn_q2_h_s_1280x720.onnx --fp16 --saveEngine=bmshj2018_hyperprior_1dn_q2_h_s_1280x720.trt --workspace=4096
trtexec --onnx=bmshj2018_hyperprior_1dn_q2_g_s_1280x720.onnx --fp16 --saveEngine=bmshj2018_hyperprior_1dn_q2_g_s_1280x720.trt --workspace=4096