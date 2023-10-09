const c = document.getElementById("M_Canvas");
const ctx = c.getContext("2d");

const N = 16; // Should be x * 2^3
const size = 17;
const padding = 2;
const colors = ["lightgray", "gray", "blue", "red", "black"];
const history_width = 1, history_height = 10;
const history_size = (size + padding) * N / history_width;

class MM {
    constructor() {
        this.data = new Array(N * N);
        this.log = new Array(N * N);
        this.history = new Array(history_size);
        this.history_head = 0;

        this.data.fill(0);
        this.log.fill(0);
        this.history.fill(0);
    }
}

var I = 0, J = 0, K = 0;
var A = new MM(), B = new MM(), C = new MM();

function draw_matrix(x, y, M, context, ei, ej) {
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            context.fillStyle = colors[M.data[i * N + j]];
            if (i == ei && j == ej)
                context.fillStyle = colors[4];
            context.fillRect(x + j * (size + padding), y + i * (size + padding), size, size);
        }
    }
}

function draw_result_matrix(x, y, M, context) {
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            context.fillStyle = colors[0];
            if (M.data[i * N + j] < N)
                context.fillRect(x + j * (size + padding), y + i * (size + padding), size, size);
            context.fillStyle = "black";
            if (M.data[i * N + j] > 0)
                context.fillRect(x + j * (size + padding), y + i * (size + padding), size, size * M.data[i * N + j] / N);
        }
    }
}

function update_indices() {
    K++;
    if (K >= N) {
        K = 0;
        J++;
        if (J >= N) {
            J = 0;
            I++;
            if (I >= N) I = 0;
        }
    }
}

function update_indices_transposed() {
    J++;
    if (J >= N) {
        J = 0;
        K++;
        if (K >= N) {
            K = 0;
            I++;
            if (I >= N) I = 0;
        }
    }
}

function reset_M(M) {
    M.data.fill(0);
    M.log.fill(0);
    M.history.fill(0);
    M.history_head = 0;
}

function reset() {
    reset_M(A);
    reset_M(B);
    reset_M(C);
    I = 0;
    J = 0;
    K = 0;
}

function tick() {
    c.width = window.innerWidth - 50;
    c.height = window.innerHeight - 50;
    T++;

    var offset = (size + padding) * N / 2;
    ctx.font = "48px courier";
    ctx.fillText("A", c.width / 2 - offset * 2.75, c.height / 2 + offset * 1.25);
    ctx.fillText("B", c.width / 2, c.height / 2 - offset * 1.5);
    ctx.fillText("C", c.width / 2, c.height / 2 + offset * 1.25);
    draw_matrix(c.width / 2 - offset * 2.5, c.height / 2 + offset * 0.25, A, ctx, I, K);
    draw_matrix(c.width / 2 + offset * 0.25, c.height / 2 - offset * 2.5, B, ctx, K, J);
    draw_result_matrix(c.width / 2 + offset * 0.25, c.height / 2 + offset * 0.25, C, ctx);

    C.data[I * N + J]++;

    if (config == 0)
        update_indices();
    if (config == 1)
        update_indices_transposed();

    if (I == 0 && J == 0 && K == 0)
        reset();
}

tick();

var T = 0;
var id = -1;
var running = 0;
var config = 0;
var INTERVAL = 100;

document.addEventListener('keypress', (event) => {
    var name = event.key;
    if (name == "r") {
        if (running == 1)
            clearInterval(id);
        running = 0;
        reset();
        tick();
    }
    if (name == "p" && running == 0) {
        // reset();
        running = 1;
        id = setInterval(tick, INTERVAL);
    } else if (name == "p" && running == 1) {
        // reset();
        running = 0;
        clearInterval(id);
    }
    if (name == "t" && config != 1) {
        config = 1;
        reset();
    }
    if (name == "d" && config != 0) {
        config = 0;
        reset();
    }
    if (name == "f") {
        INTERVAL = 16;
        if (running == 1) {
            clearInterval(id);
            id = setInterval(tick, INTERVAL);
        }
    }
    if (name == "s") {
        INTERVAL = 66;
        if (running == 1) {
            clearInterval(id);
            id = setInterval(tick, INTERVAL);
        }
    }
}, false);