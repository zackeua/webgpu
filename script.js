let gpu = {
    device: null,
    context: null,

    // compute
    pipeline: null,
    bindGroups: [],
    buffers: [],
    gridSize: 1600,
    workgroupSize: 16,
    currentBuffer: 0,

    // render
    renderPipeline: null,
    renderBindGroups: [],
};

let running = false;
let timestep = 0.01;

/* ===================== INIT ===================== */

async function initWebGPU() {
    if (!navigator.gpu) {
        console.error("WebGPU is not supported.");
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

    // ping-pong buffers
    gpu.buffers = [
        gpu.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        gpu.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
    ];

    /* ---------- COMPUTE PIPELINE ---------- */

    const computeShader = `
        @group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;

        @compute @workgroup_size(${gpu.workgroupSize}, ${gpu.workgroupSize})
        fn main(@builtin(global_invocation_id) id : vec3<u32>) {
            let x = id.x;
            let y = id.y;

            if (x >= ${gpu.gridSize}u || y >= ${gpu.gridSize}u) {
                return;
            }

            let index = y * ${gpu.gridSize}u + x;

            // Proper advection equation would go here
            let center = input[index];
            let x_gradient = input[index + max(x - 1u, 0u)] - input[index + min(x + 1u, ${gpu.gridSize}u - 1u)];
            let y_gradient = input[index + max(y - 1u, 0u) * ${gpu.gridSize}u] - input[index + min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u];

            let gradient = (x_gradient + y_gradient) * 0.5;

            let x_laplacian = input[y * ${gpu.gridSize}u + max(x - 1u, 0u)] + input[y * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)] - 2.0 * center;
            let y_laplacian = input[max(y - 1u, 0u) * ${gpu.gridSize}u + x] + input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + x] - 2.0 * center;
            let xy_laplacian = input[max(y - 1u, 0u) * ${gpu.gridSize}u + max(x - 1u, 0u)] + input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)]
                                - input[max(y - 1u, 0u) * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)]
                                - input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + max(x - 1u, 0u)];
            let yx_laplacian = input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + max(x - 1u, 0u)] + input[max(y - 1u, 0u) * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)]
                                - input[max(y - 1u, 0u) * ${gpu.gridSize}u + max(x - 1u, 0u)]
                                - input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)];

            let laplacian = 0.25 * (x_laplacian + y_laplacian + xy_laplacian + yx_laplacian);

            output[index] = center - gradient * ${timestep} + laplacian * ${timestep}*${timestep};
        }
    `;

    gpu.pipeline = gpu.device.createComputePipeline({
        layout: "auto",
        compute: {
            module: gpu.device.createShaderModule({ code: computeShader }),
            entryPoint: "main",
        },
    });

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

    /* ---------- RENDER PIPELINE ---------- */

    const renderShader = `
        struct VSOut {
            @builtin(position) pos : vec4<f32>,
            @location(0) uv : vec2<f32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) i : u32) -> VSOut {
            var pos = array<vec2<f32>, 6>(
                vec2(-1, -1),
                vec2( 1, -1),
                vec2(-1,  1),
                vec2(-1,  1),
                vec2( 1,  1),
                vec2( 1, -1)
            );

            var out : VSOut;
            out.pos = vec4(pos[i], 0, 1);
            out.uv = (pos[i] + 1.0) * 0.5;
            return out;
        }

        @group(0) @binding(0)
        var<storage, read> field : array<f32>;

        @fragment
        fn fs_main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {
            let px = u32(fragCoord.x);
            let py = u32(fragCoord.y);

            // Map canvas pixels → grid coordinates
            let gx = max(min(px * ${gpu.gridSize}u / u32(${800}), ${gpu.gridSize}u - 1u), 0u);
            let gy = max(min(py * ${gpu.gridSize}u / u32(${800}), ${gpu.gridSize}u - 1u), 0u);

            let idx = gy * ${gpu.gridSize}u + gx;
            let v = clamp(field[idx], 0.0, 1.0);

            return vec4(v, v, v, 1.0);
        }
    `;

    gpu.renderPipeline = gpu.device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: gpu.device.createShaderModule({ code: renderShader }),
            entryPoint: "vs_main",
        },
        fragment: {
            module: gpu.device.createShaderModule({ code: renderShader }),
            entryPoint: "fs_main",
            targets: [{ format }],
        },
        primitive: { topology: "triangle-list" },
    });

    gpu.renderBindGroups = [
        gpu.device.createBindGroup({
            layout: gpu.renderPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: gpu.buffers[0] } }],
        }),
        gpu.device.createBindGroup({
            layout: gpu.renderPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: gpu.buffers[1] } }],
        }),
    ];

    console.log("WebGPU initialized.");
}

function initializeField() {
    const size = gpu.gridSize;
    const data = new Float32Array(size * size);

    const cx = size * 0.5;
    const cy = size * 0.5;
    const sigma = size * 0.1;

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = x - cx;
            const dy = y - cy;

            // Gaussian blob
            const gaussian = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));

            // Sinusoidal perturbation
            const wave =
                0.25 * Math.sin(x * 0.1) * Math.cos(y * 0.1);

            data[y * size + x] = gaussian + wave;
        }
    }

    // Upload to BOTH ping-pong buffers
    gpu.device.queue.writeBuffer(gpu.buffers[0], 0, data);
    gpu.device.queue.writeBuffer(gpu.buffers[1], 0, data);
}


/* ===================== COMPUTE ===================== */

function stepSimulation() {
    const encoder = gpu.device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setPipeline(gpu.pipeline);
    pass.setBindGroup(0, gpu.bindGroups[gpu.currentBuffer]);

    const wg = gpu.gridSize / gpu.workgroupSize;
    pass.dispatchWorkgroups(wg, wg);

    pass.end();
    gpu.device.queue.submit([encoder.finish()]);

    gpu.currentBuffer = 1 - gpu.currentBuffer;
}

/* ===================== RENDER ===================== */

function renderSimulation() {
    const encoder = gpu.device.createCommandEncoder();

    const view = gpu.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view,
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store",
        }],
    });

    const renderBuffer = 1 - gpu.currentBuffer;

    pass.setPipeline(gpu.renderPipeline);
    pass.setBindGroup(0, gpu.renderBindGroups[renderBuffer]);
    pass.draw(6);

    pass.end();
    gpu.device.queue.submit([encoder.finish()]);
}

/* ===================== LOOP ===================== */

function simLoop() {
    if (!running) return;

    stepSimulation();
    renderSimulation();

    requestAnimationFrame(simLoop);
}

/* ===================== UI ===================== */

window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("render-button").addEventListener("click", async () => {
        if (!gpu.device) {
            await initWebGPU();
            initializeField();
        }

        if (!running) {
            running = true;
            requestAnimationFrame(simLoop);
        }
    });
});
