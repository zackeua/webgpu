// JavaScript code for the WebGPU demo
async function initWebGPU() {
    if (!navigator.gpu) {
        console.error("WebGPU is not supported on this browser.");
        return;
    }

    // Use webGPU to simulate the advection equation on a 2D grid
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    
    context.configure({
        device: device,
        format: format,
        alphaMode: "opaque",
    });
    const gridSize = 256;
    const bufferSize = gridSize * gridSize * 4; // 4 bytes per float
    
    // Create buffers for the simulation
    const inputBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    const outputBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    // Create a compute shader for the advection equation
    const shaderCode = `
        @group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        
        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
            let x = GlobalInvocationID.x;
            let y = GlobalInvocationID.y;
            let index = y * ${gridSize}u + x;
            
            // Simple advection logic (placeholder)
            if (x > 0u && y > 0u) {
                output[index] = input[index - 1u] * 0.99 + input[index - ${gridSize}u] * 0.01;
            } else {
                output[index] = input[index];
            }
        }
    `;
    
    const shaderModule = device.createShaderModule({ code: shaderCode });
    
    const computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: shaderModule,
            entryPoint: "main",
        },
    });
    
    const bindGroupLayout = computePipeline.getBindGroupLayout(0);
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
        ],
    });
    
    // Command encoder and compute pass
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(gridSize / 16, gridSize / 16);
    passEncoder.end();
    
    device.queue.submit([commandEncoder.finish()]);
    
    console.log("Advection simulation step executed.");
}  
window.onload = initWebGPU;