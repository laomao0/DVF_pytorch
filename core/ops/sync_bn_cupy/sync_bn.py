#!/usr/bin/env python

import torch
from torch import cuda
import cupy
import re
THREAD_BLOCK_SIZE = 256
# forward kernel

bn_forward_mean_before_allreduce_kernel = '''
    extern "C" __global__ void bn_forward_mean_before_allreduce_kernel(
        const int num,
        const int map_size,
        const int channels,
        float stat_ratio,
        const float *in,
        float *mean){
        __shared__ float buffer[256]; //THREAD_BLOCK_SIZE
        buffer[threadIdx.x] = 0;
        for (int i = threadIdx.x; i < num * map_size; i += blockDim.x){
            int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
            buffer[threadIdx.x] += in[location];
        }

        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i >>= 1){
            if (threadIdx.x < i)
            buffer[threadIdx.x] += buffer[threadIdx.x + i];
            __syncthreads();
        }

        if (threadIdx.x == 0){
            buffer[0] = buffer[0] * stat_ratio;
            mean[blockIdx.x] = buffer[0];
        }
}
'''

bn_forward_var_before_allreduce_kernel = '''
    extern "C" __global__ void bn_forward_var_before_allreduce_kernel(
        const int num,
        const int map_size,
        const int channels,
        float stat_ratio,
        const float *in,
        const float *mean,
        float *var,
        float *out){
        __shared__ float buffer[256]; //THREAD_BLOCK_SIZE
        buffer[threadIdx.x] = 0;
        for (int i = threadIdx.x; i < num * map_size; i += blockDim.x){
            int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
            out[location] = in[location] - mean[blockIdx.x];
            //buffer[threadIdx.x] += pow(out[location], (float)2);
            buffer[threadIdx.x] += out[location] * out[location];
        }
        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i >>= 1){
            if (threadIdx.x < i)
            buffer[threadIdx.x] += buffer[threadIdx.x + i];
            __syncthreads();
        }

        if (threadIdx.x == 0){
            buffer[0] = buffer[0] * stat_ratio;
            var[blockIdx.x] = buffer[0];
        }
}
'''

bn_forward_after_allreduce_kernel = '''
    extern "C" __global__ void bn_forward_after_allreduce_kernel(
        const int num,
        const int map_size,
        const int channels,
        const float stat_eps,
        const float decay,
        float *out,
        const float *mean,
        float *history_mean,
        const float *var,
        float *history_var,
        float *x_norm,
        float *x_std,
        const float *scale,
        const float *shift){

        //float temp = pow(var[blockIdx.x] + stat_eps, (float)0.5);
        float temp = sqrt(max(var[blockIdx.x], stat_eps));
        float scale_value = scale[blockIdx.x], shift_value = shift[blockIdx.x];

        for (int i = threadIdx.x; i < num * map_size; i += blockDim.x){
            int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
            x_norm[location] = out[location] / temp;
            out[location] = out[location] / temp * scale_value + shift_value;
        }
        if (threadIdx.x == 0){
            history_mean[blockIdx.x] += decay * (mean[blockIdx.x] - history_mean[blockIdx.x]);
            history_var[blockIdx.x] += decay * (var[blockIdx.x] - history_var[blockIdx.x]);
            x_std[blockIdx.x] = temp;
        }
}
'''

# backward kernel
bn_backward_before_allreduce_kernel = '''
    extern "C" __global__ void bn_backward_before_allreduce_kernel(
        const int num,
        const int map_size,
        const int channels,
        const float *in,
        const float *x_norm,
        const float *mean,
        const float *x_std,
        float *out,
        float *local_scale_diff,
        float *local_shift_diff,
        float *scale_diff,
        float *shift_diff){
        __shared__ float buffer_scale_diff[256];  //THREAD_BLOCK_SIZE
        __shared__ float buffer_shift_diff[256];  //THREAD_BLOCK_SIZE

        buffer_scale_diff[threadIdx.x] = 0;
        buffer_shift_diff[threadIdx.x] = 0;

        for (int i = threadIdx.x; i < num * map_size; i += blockDim.x){
            int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
            buffer_scale_diff[threadIdx.x] += (in[location] * x_norm[location]);
            buffer_shift_diff[threadIdx.x] += in[location];
        }
        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i >>= 1){
            if (threadIdx.x < i){
                buffer_scale_diff[threadIdx.x] += buffer_scale_diff[threadIdx.x + i];
                buffer_shift_diff[threadIdx.x] += buffer_shift_diff[threadIdx.x + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){
            local_scale_diff[blockIdx.x] = buffer_scale_diff[0];
            local_shift_diff[blockIdx.x] = buffer_shift_diff[0];
            scale_diff[blockIdx.x] += buffer_scale_diff[0];
            shift_diff[blockIdx.x] += buffer_shift_diff[0];
        }
}
'''

bn_backward_after_allreduce_kernel = '''
    extern "C" __global__ void bn_backward_after_allreduce_kernel(
        const int num,
        const int map_size,
        const int channels,
        const float *in,
        const float *x_norm,
        const float *local_scale_diff,
        const float *local_shift_diff,
        const float *scale_data,
        const float *x_std,
        float *out,
        const int num_thread){
        for (int i = threadIdx.x; i < num * map_size; i += blockDim.x){
            int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
            out[location] = scale_data[blockIdx.x] * (in[location] - (x_norm[location] * local_scale_diff[blockIdx.x] + local_shift_diff[blockIdx.x]) / (num * map_size * num_thread)) / x_std[blockIdx.x];
        }
}
'''



def cupy_kernel(strFunction, objectVariables):

    strKernel = globals()[strFunction]

    # replce the C code with real numbers
    # SIZE_0 Batch
    # SIZE_1 Channel
    # SIZE_2 H
    # SIZE_3 W
    # SIZE_x(vector), get the size of vector of dim-x
    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    # get the stride
    # VALUE_0 Batch_Stride
    # VALUE_1 Channel_Stride
    # VALUE_2 H_Stride
    # VALUE_3 W_Stride
    # VALUE_x( vector, b, c, h, w) get the value  of vector withshape BCHW
    # it return vector[ b * B_S + c * C_S + h * H_S + w * W_S ]

    while True:

        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()

        # strIndex = [B*C*H*W, C*H*W, H*W, 1]
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip(
        ) + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(
            0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel
# end


@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

@cupy.util.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
# end


def get_sizes(tensor):
    S = 1
    N = tensor.shape[0]
    C = tensor.shape[1]
    if tensor.dim() > 2:
        for i in range(2, tensor.dim()):
            S = S * tensor.shape[i]
    return N, C, S


def all_reduce_thread(input, allreduce_num, queue ):
    input_device = input.get_device()
    if input_device == 0:
        data_list = [input]
        for i in range(allreduce_num - 1):
            data_list.append(queue[i].get())

        cuda.synchronize()
        # total_sum = Synchronize.data_list[0].cpu().clone()
        # for i in range(1, Synchronize.device_num):
        #     total_sum = total_sum + Synchronize.data_list[i].cpu()

        # for i in range(0, Synchronize.device_num):
        #     with torch.cuda.device_of(Synchronize.data_list[i]):
        #         Synchronize.result_list[i] = total_sum.clone().cuda()

        cuda.nccl.all_reduce(data_list)
        cuda.synchronize()

        for i in range(allreduce_num - 1):
            queue[i].task_done()
    else:
        queue[input_device - 1].put(input)
        queue[input_device - 1].join()
    return input



class _sync_batch_norm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, momentum, eps, queue):

        allreduce_num = len(queue) + 1

        with torch.cuda.device_of(input):
            mean = input.new().resize_(input.size(1)).zero_()
            var = input.new().resize_(input.size(1)).zero_()
            x_std = input.new().resize_(input.size(1)).zero_()
            x_norm = input.new().resize_as_(input)
            output = input.new().resize_as_(input)

        assert(input.is_contiguous() == True)

        #1
        # n = output.nelement()
        N,C,HW = get_sizes(input)
        mean_ratio = float(1.0 / (HW * N * allreduce_num))
        cunnex('bn_forward_mean_before_allreduce_kernel')(
            grid=tuple([C, 1, 1]),
            block=tuple([THREAD_BLOCK_SIZE, 1, 1]),
            args=[N, HW, C, mean_ratio, input.data_ptr(), mean.data_ptr()])
        mean = all_reduce_thread(mean, allreduce_num, queue)

        #2
        N,C,HW = get_sizes(input)
        var_correction_factor = int(HW * N * allreduce_num - 1)
        if var_correction_factor == 0:
            var_correction_factor = 1
        var_ratio = float(1. / var_correction_factor)
        cunnex('bn_forward_var_before_allreduce_kernel')(
            grid=tuple([C, 1, 1]),
            block=tuple([THREAD_BLOCK_SIZE, 1, 1]),
            args=[N, HW, C, var_ratio,
                    input.data_ptr(), mean.data_ptr(), var.data_ptr(), output.data_ptr()])
        var = all_reduce_thread(var, allreduce_num, queue)

        #3
        N,C,HW = get_sizes(output)
        var_eps = float(eps)
        decay =  float(1.0 - momentum)
        cunnex('bn_forward_after_allreduce_kernel')(
            grid=tuple([C, 1, 1]),
            block=tuple([THREAD_BLOCK_SIZE, 1, 1]),
            args=[N, HW, C, var_eps, decay, output.data_ptr(), mean.data_ptr(), running_mean.data_ptr(),
                    var.data_ptr(), running_var.data_ptr(), x_norm.data_ptr(), x_std.data_ptr(),
                    weight.data_ptr(), bias.data_ptr()])

        ctx.save_for_backward(weight, bias, mean, x_norm, x_std, allreduce_num, queue)

        return output
        # end

    @staticmethod
    def backward(ctx, gradOutput):

        weight, bias, mean, x_norm, x_std, allreduce_num, queue = ctx.saved_tensors

        assert(gradOutput.is_contiguous() == True)

        with torch.cuda.device_of(grad_output):
            grad_input = grad_output.new().resize_as_(grad_output).zero_()
            grad_weight = grad_output.new().resize_as_(weight).zero_()
            grad_bias = grad_output.new().resize_as_(bias).zero_()
            grad_local_weight = grad_output.new().resize_as_(weight).zero_()
            grad_local_bias = grad_output.new().resize_as_(bias).zero_()

        N,C, HW = get_sizes(grad_output)
        cunnex('bn_backward_before_allreduce_kernel')(
            grid=tuple([C, 1, 1]),
            block=tuple([THREAD_BLOCK_SIZE, 1, 1]),
            args=[N, HW, C, grad_output.data_ptr(), x_norm.data_ptr(),
                    mean.data_ptr(), x_std.data_ptr(), grad_input.data_ptr(),
                    grad_local_weight.data_ptr(), grad_local_bias.data_ptr(),
                    grad_weight.data_ptr(), grad_bias.data_ptr()])

        grad_local_weight = all_reduce_thread(grad_local_weight, allreduce_num, queue)
        grad_local_bias = all_reduce_thread(grad_local_bias, allreduce_num, queue)

        #2
        N,C,HW = get_sizes(grad_output)
        cunnex('bn_backward_after_allreduce_kernel')(
            grid=tuple([C, 1, 1]),
            block=tuple([THREAD_BLOCK_SIZE, 1, 1]),
            args=[N, HW, C, grad_output.data_ptr(), x_norm.data_ptr(),
                grad_local_weight.data_ptr(), grad_local_bias.data_ptr(),
                weight.data_ptr(), x_std.data_ptr(), grad_input.data_ptr(),allreduce_num
                ])

        return grad_input, None, None, grad_weight, grad_bias, None, None, None
	# end
# end


def sync_batch_norm(input,
                    running_mean,
                    running_var,
                    weight=None,
                    bias=None,
                    momentum=0.1,
                    eps=1e-5,
                    queue=None):
    """Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _torch_ext.batchnormtrain:

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    res =  _sync_batch_norm.apply(input, running_mean, running_var, weight, bias, momentum, eps, queue)
    # res = _sync_batch_norm(momentum, eps, queue).apply(input, running_mean, running_var, weight, bias)
    return res
    # def FunctionChannelNorm(input1):
    # return _ChannelNorm.apply(input1)