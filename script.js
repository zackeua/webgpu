// JavaScript code for the WebGPU demo
async function initWebGPU() {
    if (!navigator.gpu) {
        console.error("WebGPU is not supported on this browser.");
        return;
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    
    context.configure({
        device: device,
        format: format,
        alphaMode: "opaque"
    });
    
    const shaderModule = device.createShaderModule({
        code: `
        @vertex
        fn vs_main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
            var pos = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 0.5),
                vec2<f32>(-0.5, -0.5),
                vec2<f32>(0.5, -0.5)
            );
            return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
        }
            
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);  // Red color
        }
        `
    });
    
    const pipeline = device.createRenderPipeline({
        vertex: {
            module: shaderModule,
            entryPoint: "vs_main"
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fs_main",
            targets: [{ format: format }]
        },
        primitive: {
            topology: "triangle-list"
        },
        layout: "auto"
    });
    
    function frame() {
        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();
        
        const renderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        };
        
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.draw(3, 1, 0, 0);
        passEncoder.end();
        
        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }
    
    requestAnimationFrame(frame);
}

window.onload = initWebGPU;