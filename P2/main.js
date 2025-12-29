"use strict";

let gl; // WebGL context
let surface; // Surface model (parabolic humming-top)
let shProgram; // Shader program
let spaceball; // Trackball rotator
let currentTime = 0.0; // For rotating light

const deg2rad = (angle) => (angle * Math.PI) / 180;

// Helper: transform point by 4x4 matrix
const transformPoint = (m, p) => {
  const x = p[0],
    y = p[1],
    z = p[2];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
};

function Model(name) {
  this.name = name;

  this.vbo = gl.createBuffer(); // vertex positions
  this.nbo = gl.createBuffer(); // vertex normals
  this.ibo = gl.createBuffer(); // indices
  this.indexCount = 0;

  /**
   * vertices: flat [x,y,z,...]
   * normals:  flat [nx,ny,nz,...]
   * indices:  flat [i0,i1,i2,...] (Uint16)
   */
  this.BufferData = (vertices, normals, indices) => {
    this.indexCount = indices.length;

    // Positions
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    // Normals
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

    // Indices
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(
      gl.ELEMENT_ARRAY_BUFFER,
      new Uint16Array(indices),
      gl.STATIC_DRAW
    );
  };

  this.Draw = () => {
    // Bind positions
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(shProgram.iAttribVertex);

    // Bind normals
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nbo);
    gl.vertexAttribPointer(shProgram.iAttribNormal, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(shProgram.iAttribNormal);

    // Bind indices and draw
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
  };
}

function ShaderProgram(name, program) {
  this.name = name;
  this.prog = program;

  this.iAttribVertex = -1;
  this.iAttribNormal = -1;

  this.iModelViewMatrix = -1;
  this.iProjectionMatrix = -1;

  this.iLightPos = -1;
  this.iAmbientColor = -1;
  this.iDiffuseColor = -1;
  this.iSpecularColor = -1;
  this.iShininess = -1;

  this.Use = () => {
    gl.useProgram(this.prog);
  };
}

/*======================  PARABOLIC HUMMING-TOP GEOMETRY  ======================*/

// parametric surface: (y is vertical axis)
// y ∈ [-h, h],  beta ∈ [0, 2π]
const parabolicHummingTopVertex = (y, beta, h, p) => {
  const rBase = Math.abs(y) - h; // |y| - h
  const r = (rBase * rBase) / (2 * p); // (|y| - h)^2 / (2p)

  const x = r * Math.cos(beta);
  const z = r * Math.sin(beta);

  return [x, y, z];
};

/**
 * Create surface mesh data for given U/V granularity.
 * uSeg: number of segments along angle (U)
 * vSeg: number of segments along vertical (V)
 *
 * Returns { positions, normals, indices }
 */
const buildGrid = (uSeg, vSeg, h, p) => {
  const positions = [];
  const normals = [];
  for (let j = 0; j <= vSeg; j++) {
    const v = j / vSeg;
    const y = -h + 2.0 * h * v; // from -h to h
    for (let i = 0; i <= uSeg; i++) {
      const u = i / uSeg;
      const beta = 2.0 * Math.PI * u;
      const [x, yy, z] = parabolicHummingTopVertex(y, beta, h, p);
      positions.push(x, yy, z);
      normals.push(0.0, 0.0, 0.0);
    }
  }
  return { positions, normals };
};

const buildIndices = (uSeg, vSeg) => {
  const indices = [];
  const vertsPerRow = uSeg + 1;
  for (let j = 0; j < vSeg; j++) {
    for (let i = 0; i < uSeg; i++) {
      const i0 = j * vertsPerRow + i;
      const i1 = i0 + 1;
      const i2 = i0 + vertsPerRow;
      const i3 = i2 + 1;
      indices.push(i0, i2, i1);
      indices.push(i1, i2, i3);
    }
  }
  return indices;
};

const accumulateNormalsFromIndices = (positions, normals, indices) => {
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t],
      i1 = indices[t + 1],
      i2 = indices[t + 2];
    const ax = positions[3 * i0],
      ay = positions[3 * i0 + 1],
      az = positions[3 * i0 + 2];
    const bx = positions[3 * i1],
      by = positions[3 * i1 + 1],
      bz = positions[3 * i1 + 2];
    const cx = positions[3 * i2],
      cy = positions[3 * i2 + 1],
      cz = positions[3 * i2 + 2];
    const ux = bx - ax,
      uy = by - ay,
      uz = bz - az;
    const vx = cx - ax,
      vy = cy - ay,
      vz = cz - az;
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    let len = Math.hypot(nx, ny, nz);
    if (len > 1e-6) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    normals[3 * i0] += nx;
    normals[3 * i0 + 1] += ny;
    normals[3 * i0 + 2] += nz;
    normals[3 * i1] += nx;
    normals[3 * i1 + 1] += ny;
    normals[3 * i1 + 2] += nz;
    normals[3 * i2] += nx;
    normals[3 * i2 + 1] += ny;
    normals[3 * i2 + 2] += nz;
  }
};

const normalizeNormals = (normals) => {
  for (let k = 0; k < normals.length; k += 3) {
    const nx = normals[k];
    const ny = normals[k + 1];
    const nz = normals[k + 2];
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-6) {
      normals[k] = nx / len;
      normals[k + 1] = ny / len;
      normals[k + 2] = nz / len;
    } else {
      normals[k] = 0.0;
      normals[k + 1] = 1.0;
      normals[k + 2] = 0.0;
    }
  }
};

const CreateSurfaceData = (uSeg = 40, vSeg = 40) => {
  const h = 1.0;
  const p = 0.5;
  const { positions, normals } = buildGrid(uSeg, vSeg, h, p);
  const indices = buildIndices(uSeg, vSeg);
  accumulateNormalsFromIndices(positions, normals, indices);
  normalizeNormals(normals);
  return { positions, normals, indices };
};

const computeProjection = () => m4.perspective(Math.PI / 8, 1, 2, 20);

const computeModelView = () => {
  const modelView = spaceball.getViewMatrix();
  const rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.7);
  const translateToPointZero = m4.translation(0, 0, -10);
  const matAccum0 = m4.multiply(rotateToPointZero, modelView);
  return m4.multiply(translateToPointZero, matAccum0);
};

const updateLightUniform = (modelViewMatrix) => {
  const lightRadius = 5.0;
  const lightHeight = 2.0;
  const lightSpeed = 0.5; // radians per second
  const angle = currentTime * lightSpeed;
  const lightPosModel = [
    lightRadius * Math.cos(angle),
    lightHeight,
    lightRadius * Math.sin(angle),
  ];
  const lightPosEye = transformPoint(modelViewMatrix, lightPosModel);
  gl.uniform3fv(shProgram.iLightPos, new Float32Array(lightPosEye));
};

const draw = () => {
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const projection = computeProjection();
  const modelViewMatrix = computeModelView();
  gl.uniformMatrix4fv(shProgram.iModelViewMatrix, false, modelViewMatrix);
  gl.uniformMatrix4fv(shProgram.iProjectionMatrix, false, projection);
  updateLightUniform(modelViewMatrix);
  surface.Draw();
};

const animate = (time) => {
  currentTime = time * 0.001; // ms → seconds
  draw();
  requestAnimationFrame(animate);
};

const validateGLResources = (prog) => {
  shProgram.iAttribVertex = gl.getAttribLocation(prog, "vertex");
  shProgram.iAttribNormal = gl.getAttribLocation(prog, "normal");
  if (shProgram.iAttribVertex < 0 || shProgram.iAttribNormal < 0) {
    throw new Error("Missing required vertex/normal attributes in shader.");
  }
  shProgram.iModelViewMatrix = gl.getUniformLocation(prog, "ModelViewMatrix");
  shProgram.iProjectionMatrix = gl.getUniformLocation(prog, "ProjectionMatrix");
  shProgram.iLightPos = gl.getUniformLocation(prog, "uLightPos");
  shProgram.iAmbientColor = gl.getUniformLocation(prog, "uAmbientColor");
  shProgram.iDiffuseColor = gl.getUniformLocation(prog, "uDiffuseColor");
  shProgram.iSpecularColor = gl.getUniformLocation(prog, "uSpecularColor");
  shProgram.iShininess = gl.getUniformLocation(prog, "uShininess");
  if (
    !shProgram.iModelViewMatrix ||
    !shProgram.iProjectionMatrix ||
    !shProgram.iLightPos
  ) {
    throw new Error("Missing essential uniforms in shader program.");
  }
};

const initGL = () => {
  const prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);
  shProgram = new ShaderProgram("Phong", prog);
  shProgram.Use();
  validateGLResources(prog);
  // New material colors for better contrast
  gl.uniform3fv(shProgram.iAmbientColor, new Float32Array([0.12, 0.1, 0.08]));
  gl.uniform3fv(shProgram.iDiffuseColor, new Float32Array([0.95, 0.55, 0.2]));
  gl.uniform3fv(shProgram.iSpecularColor, new Float32Array([0.95, 0.9, 0.8]));
  gl.uniform1f(shProgram.iShininess, 32.0);
  surface = new Model("ParabolicHummingTop");
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
};

const createProgram = (gl, vShader, fShader) => {
  const vsh = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vsh, vShader);
  gl.compileShader(vsh);
  if (!gl.getShaderParameter(vsh, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(vsh) || "Unknown vertex shader error";
    console.error("Vertex shader compilation failed:", info);
    throw new Error(
      "Vertex shader compilation failed. Check console for details."
    );
  }
  const fsh = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fsh, fShader);
  gl.compileShader(fsh);
  if (!gl.getShaderParameter(fsh, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(fsh) || "Unknown fragment shader error";
    console.error("Fragment shader compilation failed:", info);
    throw new Error(
      "Fragment shader compilation failed. Check console for details."
    );
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, vsh);
  gl.attachShader(prog, fsh);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) || "Unknown link error";
    console.error("Program link failed:", info);
    throw new Error("Shader program link failed. Check console for details.");
  }
  return prog;
};

const init = () => {
  let canvas;
  try {
    canvas = document.getElementById("webglcanvas");
    if (!canvas) {
      throw new Error("Canvas element 'webglcanvas' not found.");
    }
    gl = canvas.getContext("webgl");
    if (!gl) {
      throw new Error("WebGL not supported or disabled.");
    }
  } catch (e) {
    console.error("WebGL context initialization error:", e);
    document.getElementById("canvas-holder").innerHTML =
      "<p>WebGL unavailable. Enable hardware acceleration or use a modern browser.</p>";
    return;
  }
  try {
    initGL();
  } catch (e) {
    console.error("WebGL graphics initialization error:", e);
    document.getElementById("canvas-holder").innerHTML =
      "<p>Failed to initialize WebGL. See console logs for details.</p>";
    return;
  }

  spaceball = new TrackballRotator(canvas, draw, 0);

  // Hook up sliders for U/V granularity
  const uSlider = document.getElementById("uResolution");
  const vSlider = document.getElementById("vResolution");
  const uVal = document.getElementById("uVal");
  const vVal = document.getElementById("vVal");

  const updateSurfaceFromSliders = () => {
    const uSeg = parseInt(uSlider.value);
    const vSeg = parseInt(vSlider.value);
    uVal.textContent = uSeg.toString();
    vVal.textContent = vSeg.toString();
    const data = CreateSurfaceData(uSeg, vSeg);
    surface.BufferData(data.positions, data.normals, data.indices);
    draw();
  };

  uSlider.oninput = updateSurfaceFromSliders;
  vSlider.oninput = updateSurfaceFromSliders;
  updateSurfaceFromSliders();
  requestAnimationFrame(animate);
};
