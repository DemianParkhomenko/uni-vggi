"use strict";

let gl; // The webgl context.
let surface; // A surface model
let shProgram; // A shader program
let spaceball; // A TrackballRotator object

const deg2rad = (angle) => (angle * Math.PI) / 180;

/*======================  MODEL  ======================*/
// Wireframe surface: stores two sets of polylines – U and V.
function Model(name) {
  this.name = name;
  this.iVertexBuffer = gl.createBuffer();

  // Info for drawing each polyline: { offset, count } in vertices
  this.uLineInfo = [];
  this.vLineInfo = [];

  // Uploads all U + V lines into one buffer and remembers offsets.
  this.BufferData = (uLines, vLines) => {
    this.uLineInfo = [];
    this.vLineInfo = [];

    // Flatten all lines into one big array
    let vertices = [];
    let currentOffset = 0; // in vertices, not bytes

    // U polylines (constant y – parallels)
    for (let i = 0; i < uLines.length; i++) {
      const line = uLines[i];
      const vertCount = line.length / 3;
      this.uLineInfo.push({
        offset: currentOffset,
        count: vertCount,
      });
      vertices.push(...line);
      currentOffset += vertCount;
    }

    // V polylines (constant angle – meridians)
    for (let i = 0; i < vLines.length; i++) {
      const line = vLines[i];
      const vertCount = line.length / 3;
      this.vLineInfo.push({
        offset: currentOffset,
        count: vertCount,
      });
      vertices.push(...line);
      currentOffset += vertCount;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  };

  this.Draw = () => {
    gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
    gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(shProgram.iAttribVertex);

    // Draw U polylines (orange)
    gl.uniform4fv(shProgram.iColor, [1.0, 0.6, 0.0, 1.0]);
    for (let i = 0; i < this.uLineInfo.length; i++) {
      const info = this.uLineInfo[i];
      gl.drawArrays(gl.LINE_STRIP, info.offset, info.count);
    }

    // Draw V polylines (sky blue)
    gl.uniform4fv(shProgram.iColor, [0.2, 0.7, 1.0, 1.0]);
    for (let i = 0; i < this.vLineInfo.length; i++) {
      const info = this.vLineInfo[i];
      gl.drawArrays(gl.LINE_STRIP, info.offset, info.count);
    }
  };
}

/*======================  SHADER PROGRAM  ======================*/
function ShaderProgram(name, program) {
  this.name = name;
  this.prog = program;

  this.iAttribVertex = -1;
  this.iColor = -1;
  this.iModelViewProjectionMatrix = -1;

  this.Use = () => {
    gl.useProgram(this.prog);
  };
}

/*======================  DRAW  ======================*/
const computeTransforms = (modelView) => {
  const rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.7);
  const translateToPointZero = m4.translation(0, 0, -10);
  const matAccum0 = m4.multiply(rotateToPointZero, modelView);
  return m4.multiply(translateToPointZero, matAccum0);
};

const computeMVP = (modelView) => {
  const projection = m4.perspective(Math.PI / 8, 1, 8, 12);
  const mv = computeTransforms(modelView);
  return m4.multiply(projection, mv);
};

const draw = () => {
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const modelView = spaceball.getViewMatrix();
  const modelViewProjection = computeMVP(modelView);

  gl.uniformMatrix4fv(
    shProgram.iModelViewProjectionMatrix,
    false,
    modelViewProjection
  );

  surface.Draw();
};

/*======================  GEOMETRY  ======================*/

// Parabolic Humming-Top parametric function, mapped to (x, y, z)
// y ∈ [-h, h], beta ∈ [0, 2π]
const parabolicHummingTopVertex = (y, beta, h, p) => {
  const rBase = Math.abs(y) - h; // |y| - h
  const r = (rBase * rBase) / (2 * p); // (|y| - h)^2 / (2p)

  const x = r * Math.cos(beta);
  const z = r * Math.sin(beta);

  return [x, y, z];
};

const generateULines = (h, p, vSegments, uSegments) => {
  const uLines = [];
  for (let j = 0; j <= vSegments; j++) {
    const y = -h + (2 * h * j) / vSegments; // from -h to h
    const line = [];
    for (let i = 0; i <= uSegments; i++) {
      const beta = (2 * Math.PI * i) / uSegments;
      const [x, yy, z] = parabolicHummingTopVertex(y, beta, h, p);
      line.push(x, yy, z);
    }
    uLines.push(line);
  }
  return uLines;
};

const generateVLines = (h, p, vSegments, uSegments) => {
  const vLines = [];
  for (let i = 0; i <= uSegments; i++) {
    const beta = (2 * Math.PI * i) / uSegments;
    const line = [];
    for (let j = 0; j <= vSegments; j++) {
      const y = -h + (2 * h * j) / vSegments; // from -h to h
      const [x, yy, z] = parabolicHummingTopVertex(y, beta, h, p);
      line.push(x, yy, z);
    }
    vLines.push(line);
  }
  return vLines;
};

const CreateSurfaceData = () => {
  // geometric parameters
  const h = 1.0; // height of one sheet
  const p = 0.5; // parabola parameter

  // grid resolution
  const vSegments = 40; // along y (vertical)
  const uSegments = 64; // angle segments

  const uLines = generateULines(h, p, vSegments, uSegments);
  const vLines = generateVLines(h, p, vSegments, uSegments);
  return { uLines, vLines };
};

const validateGLResources = (prog) => {
  shProgram.iAttribVertex = gl.getAttribLocation(prog, "vertex");
  if (shProgram.iAttribVertex < 0) {
    throw new Error("Missing vertex attribute 'vertex' in shader program.");
  }

  shProgram.iModelViewProjectionMatrix = gl.getUniformLocation(
    prog,
    "ModelViewProjectionMatrix"
  );
  if (!shProgram.iModelViewProjectionMatrix) {
    throw new Error(
      "Missing uniform 'ModelViewProjectionMatrix' in shader program."
    );
  }

  shProgram.iColor = gl.getUniformLocation(prog, "color");
  if (!shProgram.iColor) {
    throw new Error("Missing uniform 'color' in shader program.");
  }
};

const initGL = () => {
  const prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);

  shProgram = new ShaderProgram("Basic", prog);
  shProgram.Use();

  validateGLResources(prog);

  // Create and fill surface model
  const surfaceData = CreateSurfaceData();
  surface = new Model("ParabolicHummingTop");
  surface.BufferData(surfaceData.uLines, surfaceData.vLines);

  gl.enable(gl.DEPTH_TEST);
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

/*======================  INIT  ======================*/
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
    initGL(); // initialize the WebGL graphics context
  } catch (e) {
    console.error("WebGL graphics initialization error:", e);
    document.getElementById("canvas-holder").innerHTML =
      "<p>Failed to initialize WebGL. See console logs for details.</p>";
    return;
  }

  spaceball = new TrackballRotator(canvas, draw, 0);
  draw();
};
