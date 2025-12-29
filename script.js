let gpu = {
    device: null,
    context: null,
    pipeline: null,
    bindGroups: [],
    buffers: [],
    gridSize: 256,
    workgroupSize: 16,
    currentBuffer: 0
};
let running = false;
let timestep = 0.01;

async function initWebGPU() {
    if (!navigator.gpu) {
        console.error("WebGPU is not supported on this browser.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    gpu.device = await adapter.requestDevice();

    const canvas = document.getElementById("webgpu-canvas");
    gpu.context = canvas.getContext("webgpu");

    const format = navigator.gpu.getPreferredCanvasFormat();
    gpu.context.configure({
        device: gpu.device,
        format,
        alphaMode: "opaque",
    });

    const bufferSize = gpu.gridSize * gpu.gridSize * 4;

    // Create ping-pong buffers
    gpu.buffers = [
        gpu.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
        gpu.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
    ];

    const shaderCode = `
        @group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;

        @compute @workgroup_size(${gpu.workgroupSize}, ${gpu.workgroupSize})
        fn main(@builtin(global_invocation_id) id : vec3<u32>) {
            let x = id.x;
            let y = id.y;
            let index = y * ${gpu.gridSize}u + x;

            if (x > 0u && y > 0u) {
                output[index] =
                    input[index - 1u] * 0.99 +
                    input[index - ${gpu.gridSize}u] * 0.01;
            } else {
                output[index] = input[index];
            }
        }
    `;

    const shaderModule = gpu.device.createShaderModule({ code: shaderCode });

    gpu.pipeline = gpu.device.createComputePipeline({
        layout: "auto",
        compute: {
            module: shaderModule,
            entryPoint: "main",
        },
    });

    // Create bind groups for both buffer directions
    gpu.bindGroups = [
        gpu.device.createBindGroup({
            layout: gpu.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: gpu.buffers[0] } },
                { binding: 1, resource: { buffer: gpu.buffers[1] } },
            ],
        }),
        gpu.device.createBindGroup({
            layout: gpu.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: gpu.buffers[1] } },
                { binding: 1, resource: { buffer: gpu.buffers[0] } },
            ],
        }),
    ];

    console.log("WebGPU initialized.");
}

function stepSimulation() {
    const commandEncoder = gpu.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();

    pass.setPipeline(gpu.pipeline);
    pass.setBindGroup(0, gpu.bindGroups[gpu.currentBuffer]);

    const workgroups = gpu.gridSize / gpu.workgroupSize;
    pass.dispatchWorkgroups(workgroups, workgroups);

    pass.end();
    gpu.device.queue.submit([commandEncoder.finish()]);

    // Swap buffers
    gpu.currentBuffer = 1 - gpu.currentBuffer;
}

function animate() {
    stepSimulation();
    requestAnimationFrame(animate);
}

window.addEventListener("DOMContentLoaded", () => {
    const renderButton = document.getElementById("render-button");
    const timestepInput = document.getElementById("timestep-size");
    const simulationSelect = document.getElementById("simulation-select");

    timestep = parseFloat(timestepInput.value);

    timestepInput.addEventListener("input", () => {
        timestep = parseFloat(timestepInput.value);
    });

    simulationSelect.addEventListener("change", (e) => {
        selectSimulation(e.target.value);
    });

    renderButton.addEventListener("click", async () => {
        if (!gpu.device) {
            await initWebGPU();
        }

        if (!running) {
            running = true;
            requestAnimationFrame(simLoop);
        }
    });
});


function simLoop() {
    if (!running) return;

    stepSimulation(timestep);

    requestAnimationFrame(simLoop);
}


function stepSimulation(dt) {
    const commandEncoder = gpu.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();

    pass.setPipeline(gpu.pipeline);
    pass.setBindGroup(0, gpu.bindGroups[gpu.currentBuffer]);

    const workgroups = gpu.gridSize / gpu.workgroupSize;
    pass.dispatchWorkgroups(workgroups, workgroups);

    pass.end();
    gpu.device.queue.submit([commandEncoder.finish()]);

    gpu.currentBuffer = 1 - gpu.currentBuffer;
}
