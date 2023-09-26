const c = document.getElementById("M_Canvas");
const ctx = c.getContext("2d");

const INTERVAL = 25;
const N = 16; // Should be x * 2^3
const size = 12;
const padding = 2;
const colors = ["lightgray", "gray", "blue", "red", "black"];
const history_width = 1, history_height = 10;
const history_size = (size + padding) * N / history_width;

var M1_log = [], M1 = [], M2_log = [], M2 = [];
var M1_L = [0, 0, 0], M2_L = [0, 0, 0];

var M3 = [];

var h1_head = 0, h2_head = 0;
var M1_history = [], M2_history = [];
for (var i = 0; i < 100; i++) {
    M1_history.push(0);
    M2_history.push(0);
}

function init_matrix(M) {
    for (var i = 0; i < N; i++) {
        M.push([]);
        for (var j = 0; j < N; j++) {
            M[i].push(0);
        }
    }
}

function clear_matrix(M) {
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            M[i][j] = 0;
        }
    }
}

init_matrix(M1_log);
init_matrix(M1);

init_matrix(M2_log);
init_matrix(M2);

init_matrix(M3);

var I = 0, J = 0, K = 0;

var T = 0;
var id = setInterval(tick, INTERVAL);

function find_min(M, M_log, level) {
    var min_T = T, min_i = 0, min_j = 0;
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            if (M[i][j] >= level && M_log[i][j] < min_T) {
                min_T = M_log[i][j];
                min_i = i;
                min_j = j;
            }
        }
    }
    return [min_i, min_j];
}

function evict(M, M_log, M_L, i, j) {
    if (M[i][j] < 3) M_L[0]++;
    if (M[i][j] < 2) M_L[1]++;
    if (M[i][j] < 1) M_L[2]++;

    M[i][j] = 3;

    if (M_L[0] > L1) {
        var [min_i, min_j] = find_min(M, M_log, 3);
        M[min_i][min_j] = 2;
        M_L[0]--;
    }
    if (M_L[1] > L2) {
        var [min_i, min_j] = find_min(M, M_log, 2);
        M[min_i][min_j] = 1;
        M_L[1]--;
    }
    if (M_L[2] > L3) {
        var [min_i, min_j] = find_min(M, M_log, 1);
        M[min_i][min_j] = 0;
        M_L[2]--;
    }
}

const L3 = (N * N) / 4, L2 = L3 / 4, L1 = L2 / 4;
var L3I = [0, 0, 0];
var L2I = [0, 0, 0];
var L1I = [0, 0, 0];
var indices = [0, 0, 0];

function advance(index, limit, order, i, d) {
    if (d > 2) return 1;

    index[i]++;

    if (index[i] >= limit[i]) {
        index[i] = 0;

        return advance(index, limit, order, order[d + 1], d + 1);
    }

    return 0;
}

function next() {
    var f = advance(indices, [N, N, N], [2, 1, 0], 2, 0);
    if (f == 1) {
        M3 = [];
        init_matrix(M3);
    }
    I = indices[0];
    J = indices[1];
    K = indices[2];
}

function next_block_L3() {
    var f = advance(indices, [N / 2, N / 2, N / 2], [2, 1, 0], 2, 0);
    if (f == 1) {
        var f3 = advance(L3I, [2, 2, 2], [1, 2, 0], 1, 0);
        if (f3 == 1) {
            M3 = [];
            init_matrix(M3);
        }
    }
    I = L3I[0] * N / 2 + indices[0];
    J = L3I[1] * N / 2 + indices[1];
    K = L3I[2] * N / 2 + indices[2];
}

function next_block_L2() {
    var f = advance(indices, [N / 4, N / 4, N / 4], [2, 1, 0], 2, 0);
    if (f == 1) {
        var f2 = advance(L2I, [2, 2, 2], [2, 1, 0], 2, 0);
        if (f2 == 1) {
            var f3 = advance(L3I, [2, 2, 2], [2, 1, 0], 2, 0);

            if (f3 == 1) {
                M3 = [];
                init_matrix(M3);
            }
        }
    }
    I = L3I[0] * N / 2 + L2I[0] * N / 4 + indices[0];
    J = L3I[1] * N / 2 + L2I[1] * N / 4 + indices[1];
    K = L3I[2] * N / 2 + L2I[2] * N / 4 + indices[2];
}

function next_block_L1() {
    var f = advance(indices, [N / 8, N / 8, N / 8], [2, 1, 0], 2, 0);
    if (f == 1) {
        var f1 = advance(L1I, [2, 2, 2], [2, 1, 0], 2, 0);
        if (f1 == 1) {
            var f2 = advance(L2I, [2, 2, 2], [2, 1, 0], 2, 0);

            if (f2 == 1) {
                var f3 = advance(L3I, [2, 2, 2], [2, 1, 0], 2, 0);

                if (f3 == 1) {
                    M3 = [];
                    init_matrix(M3);
                }
            }
        }
    }
    I = L3I[0] * N / 2 + L2I[0] * N / 4 + L1I[0] * N / 8 + indices[0];
    J = L3I[1] * N / 2 + L2I[1] * N / 4 + L1I[1] * N / 8 + indices[1];
    K = L3I[2] * N / 2 + L2I[2] * N / 4 + L1I[2] * N / 8 + indices[2];
}

var offScreenCanvas = document.createElement('canvas');

var config = 0;

function tick() {
    T++;
    M1_log[I][K] = T;
    M2_log[K][J] = T;
    M3[I][J]++;

    M1_history[h1_head] = M1[I][K];
    h1_head++;
    if (h1_head >= history_size) h1_head = 0;

    M2_history[h2_head] = M2[K][J];
    h2_head++;
    if (h2_head >= history_size) h2_head = 0;

    evict(M1, M1_log, M1_L, I, K);
    evict(M2, M2_log, M2_L, K, J);

    if (config == 0)
        next();
    else if (config == 1)
        next_block_L3();
    else if (config == 2)
        next_block_L2();
    else if (config == 3)
        next_block_L1();

    ctx.canvas.width = window.innerWidth - 50;
    ctx.canvas.height = window.innerHeight - 50;

    offScreenCanvas.width = ctx.canvas.width;
    offScreenCanvas.height = ctx.canvas.height;
    var context = offScreenCanvas.getContext("2d");

    const offset = (N + 3) * (size + padding) / 2;

    draw_matrix(-offset, offset, M1, context, I, K);
    draw_matrix(offset, -offset - history_height * 3, M2, context, K, J);
    draw_result_matrix(offset, offset, M3, context);

    draw_history(-offset, offset + (size + padding) * (N + 1), M1_history, h1_head, context);
    draw_history(offset, -offset - history_height * 3 + (size + padding) * (N + 1), M2_history, h2_head, context);

    ctx.drawImage(offScreenCanvas, 0, 0);
}

function draw_matrix(x, y, M, context, I, J) {
    x += context.canvas.width / 2 - N * (size + padding) / 2;
    y += context.canvas.height / 2 - N * (size + padding) / 2;

    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            context.fillStyle = colors[M[i][j]];
            if (i == I && j == J)
                context.fillStyle = colors[4];
            context.fillRect(x + j * (size + padding), y + i * (size + padding), size, size);
        }
    }
}

function draw_result_matrix(x, y, M, context) {
    x += context.canvas.width / 2 - N * (size + padding) / 2;
    y += context.canvas.height / 2 - N * (size + padding) / 2;

    context.fillStyle = "black";
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            if (M[i][j] > 0)
                context.fillRect(x + j * (size + padding), y + i * (size + padding), size, size * M[i][j] / N);
        }
    }
}

function draw_history(x, y, H, p, context) {
    x += context.canvas.width / 2 - N * (size + padding) / 2;
    y += context.canvas.height / 2 - N * (size + padding) / 2;

    var i = p + 1;
    if (i >= history_size) i = 0;
    var t = 0;

    while (i != p) {
        context.fillStyle = colors[3];
        if (H[i] == 3)
            context.fillRect(x + t * history_width, y, history_width, history_height);
        context.fillStyle = colors[2];
        if (H[i] >= 2)
            context.fillRect(x + t * history_width, y + history_height, history_width, history_height);
        context.fillStyle = colors[1];
        if (H[i] >= 1)
            context.fillRect(x + t * history_width, y + history_height * 2, history_width, history_height);
        context.fillStyle = colors[0];
        if (H[i] == 0)
            context.fillRect(x + t * history_width, y + history_height * 3, history_width, history_height);
        t++;
        i++;
        if (i >= history_size) i = 0;
    }
}

document.addEventListener('keypress', (event) => {
    var name = event.key;
    var any = 0;
    if (name == "a") {
        config++;
        any = 1;
    }

    if (any == 1) {
        clear_matrix(M1);
        clear_matrix(M2);
        clear_matrix(M3);
        M1_L = [0, 0, 0];
        M2_L = [0, 0, 0];

        I = 0, J = 0, K = 0;
        indices = [0, 0, 0];
        L1I = [0, 0, 0];
        L2I = [0, 0, 0];
        L3I = [0, 0, 0];
    }
}, false);