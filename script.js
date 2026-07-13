let gpu = {
    device: null,
    context: null,

    // compute
    pipeline: null,
    bindGroups: [],
    buffers: [],
    uniformBuffer: null,
    simulation: "Advection equation",
    gridSize: 2**11, // 2048
    workgroupSize: 16,
    currentBuffer: 0,

    // render
    renderPipeline: null,
    renderBindGroups: [],
};

let running = false;
let lastFrameTime = 0;
let timestep = 0.0001;
let diffusion = 0.0002;
let velocityX = 0.2;
let velocityY = 0.05;
let golDensity = 0.5;
let isingTemperature = 2.5;
let isingField = 0.0;

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

    gpu.uniformBuffer = gpu.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const updateComputeParams = () => {
        const params = new Float32Array([
            timestep,
            gpu.simulation === "Advection equation" ? diffusion : gpu.simulation === "Icing model" ? icingDeposition : 0.0,
            gpu.simulation === "Advection equation" ? velocityX : gpu.simulation === "Icing model" ? icingMelt : 0.0,
            gpu.simulation === "Advection equation" ? velocityY : 0.0,
        ]);
        gpu.device.queue.writeBuffer(gpu.uniformBuffer, 0, params);
    };
    updateComputeParams();

    const createComputePipeline = () => {
        const advectionComputeShader = `
            @group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read_write> output : array<f32>;
            @group(0) @binding(2) var<uniform> params : vec4<f32>;

            @compute @workgroup_size(${gpu.workgroupSize}, ${gpu.workgroupSize})
            fn main(@builtin(global_invocation_id) id : vec3<u32>) {
                let x = id.x;
                let y = id.y;

                if (x >= ${gpu.gridSize}u || y >= ${gpu.gridSize}u) {
                    return;
                }

                let index = y * ${gpu.gridSize}u + x;
                let center = input[index];

                var left = center;
                if (x > 0u) {
                    left = input[index - 1u];
                }

                var right = center;
                if (x < ${gpu.gridSize}u - 1u) {
                    right = input[index + 1u];
                }

                var up = center;
                if (y > 0u) {
                    up = input[index - ${gpu.gridSize}u];
                }

                var down = center;
                if (y < ${gpu.gridSize}u - 1u) {
                    down = input[index + ${gpu.gridSize}u];
                }

                let gridSizeF = f32(${gpu.gridSize}u);
                let invDx = gridSizeF;
                let invDx2 = gridSizeF * gridSizeF;

                let dt = params.x;
                let nu = params.y;
                let vx = params.z;
                let vy = params.w;

                let u_x = 0.5 * (right - left) * invDx;
                let u_y = 0.5 * (down - up) * invDx;

                let u_xx = (right - 2.0 * center + left) * invDx2;
                let u_yy = (down - 2.0 * center + up) * invDx2;

                let advective = -(vx * u_x + vy * u_y);
                let diffusive = nu * (u_xx + u_yy);

                output[index] = clamp(center + (advective + diffusive) * dt, 0.0, 1.0);
            }
        `;

        const gameComputeShader = `
            @group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read_write> output : array<f32>;
            @group(0) @binding(2) var<uniform> params : vec4<f32>;

            @compute @workgroup_size(${gpu.workgroupSize}, ${gpu.workgroupSize})
            fn main(@builtin(global_invocation_id) id : vec3<u32>) {
                let x = id.x;
                let y = id.y;
                if (x >= ${gpu.gridSize}u || y >= ${gpu.gridSize}u) {
                    return;
                }
                let index = y * ${gpu.gridSize}u + x;

                var neighbors = 0u;
                neighbors += u32(input[y * ${gpu.gridSize}u + max(x - 1u, 0u)]);
                neighbors += u32(input[y * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)]);
                neighbors += u32(input[max(y - 1u, 0u) * ${gpu.gridSize}u + x]);
                neighbors += u32(input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + x]);
                neighbors += u32(input[max(y - 1u, 0u) * ${gpu.gridSize}u + max(x - 1u, 0u)]);
                neighbors += u32(input[max(y - 1u, 0u) * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)]);
                neighbors += u32(input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + max(x - 1u, 0u)]);
                neighbors += u32(input[min(y + 1u, ${gpu.gridSize}u - 1u) * ${gpu.gridSize}u + min(x + 1u, ${gpu.gridSize}u - 1u)]);

                if (u32(input[index]) > 0u) {
                    if (neighbors < 2u || neighbors > 3u) {
                        output[index] = 0.0;
                    } else {
                        output[index] = 1.0;
                    }
                } else {
                    output[index] = input[index];
                }
            }
        `;

        const isingComputeShader = `
            @group(0) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(1) var<storage, read_write> output : array<f32>;
            @group(0) @binding(2) var<uniform> params : vec4<f32>;

            fn rand(x : f32) -> f32 {
                return fract(sin(x) * 43758.5453123);
            }

            @compute @workgroup_size(${gpu.workgroupSize}, ${gpu.workgroupSize})
            fn main(@builtin(global_invocation_id) id : vec3<u32>) {
                let x = id.x;
                let y = id.y;
                if (x >= ${gpu.gridSize}u || y >= ${gpu.gridSize}u) {
                    return;
                }
                let index = y * ${gpu.gridSize}u + x;
                let center = input[index];

                var left = center;
                if (x > 0u) {
                    left = input[index - 1u];
                }
                var right = center;
                if (x < ${gpu.gridSize}u - 1u) {
                    right = input[index + 1u];
                }
                var up = center;
                if (y > 0u) {
                    up = input[index - ${gpu.gridSize}u];
                }
                var down = center;
                if (y < ${gpu.gridSize}u - 1u) {
                    down = input[index + ${gpu.gridSize}u];
                }

                let temperature = max(params.y, 0.01);
                let field = params.z;
                let seed = f32(index) * 12.9898 + params.x * 78.233 + f32(y) * 5.17;
                let r = rand(seed);

                let neighborSum = left + right + up + down;
                let deltaE = 2.0 * center * (neighborSum + field);
                let accept = deltaE <= 0.0 || exp(-deltaE / temperature) > r;

                output[index] = select(center, -center, accept);
            }
        `;

        const computeShader = gpu.simulation === "Advection equation" ? advectionComputeShader :
                              gpu.simulation === "Game of Life" ? gameComputeShader :
                              isingComputeShader;

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
                    { binding: 2, resource: { buffer: gpu.uniformBuffer } },
                ],
            }),
            gpu.device.createBindGroup({
                layout: gpu.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: gpu.buffers[1] } },
                    { binding: 1, resource: { buffer: gpu.buffers[0] } },
                    { binding: 2, resource: { buffer: gpu.uniformBuffer } },
                ],
            }),
        ];
    };

    createComputePipeline();

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

    if (gpu.simulation === "Advection equation") {
        console.log("Initializing field for Advection equation.");
        
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
                const wave = 0.25 * Math.sin(x * 0.1) * Math.cos(y * 0.1);

                data[y * size + x] = gaussian + wave;
            }
        }
    } else if (gpu.simulation === "Game of Life") {
        console.log("Initializing field for Game of Life.");
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Random noise initialization
                data[y * size + x] = Math.random() > 0.5 ? 1.0 : 0.0;
            }
        }
    } else if (gpu.simulation === "Ising model") {
        console.log("Initializing field for Ising model.");
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                data[y * size + x] = Math.random() > 0.5 ? 1.0 : -1.0;
            }
        }
    } else {
        console.log("Initializing field for default simulation.");
    }

    // Upload to BOTH ping-pong buffers
    gpu.device.queue.writeBuffer(gpu.buffers[0], 0, data);
    gpu.device.queue.writeBuffer(gpu.buffers[1], 0, data);
}

function setSimulation(option) {
    console.log("Selected simulation:", option);

    if (option === "Advection equation") {
        gpu.simulation = "Advection equation";
    } else if (option === "Game of Life") {
        gpu.simulation = "Game of Life";
    } else if (option === "Ising model") {
        gpu.simulation = "Ising model";
    } else {
        console.warn("Unknown simulation option:", option, "defaulting to Advection equation.");
        gpu.simulation = "Advection equation";
    }
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

function simLoop(timestamp) {
    if (!running) return;

    const elapsed = timestamp - lastFrameTime;
    const minFrameTime = gpu.simulation === "Advection equation" ? 16 : 120;

    if (elapsed >= minFrameTime) {
        lastFrameTime = timestamp;
        stepSimulation();
        renderSimulation();
    }

    requestAnimationFrame(simLoop);
}

/* ===================== UI ===================== */

window.addEventListener("DOMContentLoaded", () => {
    const tabButtons = document.querySelectorAll(".tab");
    const configs = {
        "Advection equation": document.getElementById("config-advection"),
        "Game of Life": document.getElementById("config-gol"),
        "Ising model": document.getElementById("config-ising"),
    };

    const setActiveSimulation = (simulation) => {
        gpu.simulation = simulation;

        tabButtons.forEach((button) => {
            const active = button.dataset.simulation === simulation;
            button.classList.toggle("active", active);
        });

        Object.values(configs).forEach((config) => config.classList.remove("active"));
        configs[simulation].classList.add("active");

        const status = document.getElementById("status");
        status.textContent = `Selected: ${simulation}`;
    };

    tabButtons.forEach((button) => {
        button.addEventListener("click", () => {
            setActiveSimulation(button.dataset.simulation);
        });
    });

    const updateParams = () => {
        if (gpu.simulation === "Advection equation") {
            timestep = parseFloat(document.getElementById("advection-timestep").value) || timestep;
            diffusion = parseFloat(document.getElementById("diffusion").value) || diffusion;
            velocityX = parseFloat(document.getElementById("velocity-x").value) || velocityX;
            velocityY = parseFloat(document.getElementById("velocity-y").value) || velocityY;
        } else if (gpu.simulation === "Game of Life") {
            timestep = parseFloat(document.getElementById("gol-timestep").value) || timestep;
            golDensity = parseFloat(document.getElementById("gol-density").value) || golDensity;
        } else if (gpu.simulation === "Ising model") {
            timestep = parseFloat(document.getElementById("ising-timestep").value) || timestep;
            isingTemperature = parseFloat(document.getElementById("ising-temperature").value) || isingTemperature;
            isingField = parseFloat(document.getElementById("ising-field").value) || isingField;
        }

        if (gpu.device && gpu.uniformBuffer) {
            const params = new Float32Array([
                timestep,
                gpu.simulation === "Advection equation" ? diffusion : gpu.simulation === "Ising model" ? isingTemperature : 0.0,
                gpu.simulation === "Advection equation" ? velocityX : gpu.simulation === "Ising model" ? isingField : 0.0,
                gpu.simulation === "Advection equation" ? velocityY : 0.0,
            ]);
            gpu.device.queue.writeBuffer(gpu.uniformBuffer, 0, params);
        }
    };

    document.querySelectorAll("input[type='number']").forEach((input) => {
        input.addEventListener("input", updateParams);
    });

    document.getElementById("render-button").addEventListener("click", async () => {
        updateParams();

        if (!gpu.device) {
            await initWebGPU();
            initializeField();
        }

        if (!running) {
            running = true;
            requestAnimationFrame(simLoop);
        }

        const status = document.getElementById("status");
        status.textContent = `Running ${gpu.simulation}`;
    });

    setActiveSimulation(gpu.simulation);
});
