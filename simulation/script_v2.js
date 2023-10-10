const c = document.getElementById("M_Canvas");
const ctx = c.getContext("2d");

const N = 16; // Should be x * 2^3
const size = 19;
const padding = 2;
const colors = ["lightgray", "gray", "blue", "red", "black"];
const history_width = 1, history_height = 10;
const history_size = (size + padding) * N / history_width;
var L3 = N * N / 4;
var L2 = L3 / 4;
var L1 = L2 / 4;

class MM {
    constructor() {
        this.data = new Array(N * N);
        this.history = new Array(history_size);
        this.history_head = 0;

        this.data.fill(0);
        this.history.fill(0);
    }
}

var I = 0, J = 0, K = 0;
var A = new MM(), B = new MM(), C = new MM();

function draw_matrix(x, y, M, context, ei, ej) {
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            if (L1s == 1 && L3 - M.data[i * N + j] < L1)
                context.fillStyle = colors[3];
            else if (L2s == 1 && L3 - M.data[i * N + j] < L2)
                context.fillStyle = colors[2];
            else if (L3s == 1 && M.data[i * N + j] > 0)
                context.fillStyle = colors[1];
            else
                context.fillStyle = colors[0];

            if (i == ei && j == ej)
                context.fillStyle = colors[4];
            context.fillRect(x + j * (size + padding), y + i * (size + padding), size, size);
        }
    }
}

function draw_history(x, y, M, context) {
    var i = M.history_head + 1;
    if (i >= history_size) i = 0;
    var t = 0;

    ctx.font = "12px courier";
    ctx.fillStyle = "black";
    if (L1s == 1)
        ctx.fillText("L1", x + 2 + history_width * history_size, y + history_height);
    if (L2s == 1)
        ctx.fillText("L2", x + 2 + history_width * history_size, y + history_height * 2);
    if (L3s == 1)
        ctx.fillText("L3", x + 2 + history_width * history_size, y + history_height * 3);
    if (L1s + L2s + L3s >= 1)
        ctx.fillText("Miss", x + 2 + history_width * history_size, y + history_height * 4);

    while (i != M.history_head) {
        var any = 0;
        if (L1s == 1 && M.history[i] == 1) {
            any = 1;
            context.fillStyle = colors[3];
            context.fillRect(x + t * history_width, y, history_width, history_height);
        }
        if (L2s == 1 && M.history[i] >= 1 && M.history[i] <= 2) {
            any = 1;
            context.fillStyle = colors[2];
            context.fillRect(x + t * history_width, y + history_height, history_width, history_height);
        }
        if (L3s == 1 && M.history[i] >= 1 && M.history[i] <= 3) {
            any = 1;
            context.fillStyle = colors[1];
            context.fillRect(x + t * history_width, y + history_height * 2, history_width, history_height);
        }

        if (L1s + L2s + L3s >= 1 && any == 0 && M.history[i] > 0) {
            context.fillStyle = colors[0];
            context.fillRect(x + t * history_width, y + history_height * 3, history_width, history_height);
        }

        t++;
        i++;
        if (i >= history_size) i = 0;
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

function update_indices(N, step) {
    K += step;
    if (K % N == 0) {
        K -= N;
        J += step;
        if (J % N == 0) {
            J -= N;
            I += step;
            if (I % N == 0) I -= N;
        }
    }
}

function update_indices_transposed(N, step) {
    J += step;
    if (J % N == 0) {
        J -= N;
        K += step;
        if (K % N == 0) {
            K -= N;
            I += step;
            if (I % N == 0) I -= N;
        }
    }
}

function update_indices_L3block(N) {
    update_indices(N / 2, 1);
    if ((I % (N / 2)) + (J % (N / 2)) + (K % (N / 2)) == 0) {
        update_indices(N, N / 2);
    }
}

function update_indices_L2block(N) {
    update_indices(N / 4, 1);
    if ((I % (N / 4)) + (J % (N / 4)) + (K % (N / 4)) == 0) {
        update_indices(N / 2, N / 4);
        if ((I % (N / 2)) + (J % (N / 2)) + (K % (N / 2)) == 0) {
            update_indices(N, N / 2);
        }
    }
}

function update_indices_L1block(N) {
    update_indices(N / 8, 1);
    if ((I % (N / 8)) + (J % (N / 8)) + (K % (N / 8)) == 0) {
        update_indices(N / 4, N / 8);
        if ((I % (N / 4)) + (J % (N / 4)) + (K % (N / 4)) == 0) {
            update_indices(N / 2, N / 4);
            if ((I % (N / 2)) + (J % (N / 2)) + (K % (N / 2)) == 0) {
                update_indices(N, N / 2);
            }
        }
    }
}

function update_indices_OpenBLAS() {
    J++;
    if (J % 2 == 0) {
        J -= 2;
        K++;
        if (K % (N / 2) == 0) {
            K -= N / 2;
            I++;
            if (I % (N / 2) == 0) {
                I -= N / 2;
                J += 2;
                if (J >= N) {
                    J = 0;
                    K += N / 2;
                    if (K >= N) {
                        K = 0;
                        I += N / 2;
                        if (I >= N) I = 0;
                    }
                }
            }
        }
    }
}

function update_M(M, mi, mj) {
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
            if (M.data[i * N + j] > 0 && M.data[i * N + j] > M.data[mi * N + mj])
                M.data[i * N + j]--;
        }
    }
    if (L3 - M.data[mi * N + mj] < L1)
        M.history[M.history_head] = 1;
    else if (L3 - M.data[mi * N + mj] < L2)
        M.history[M.history_head] = 2;
    else if (M.data[mi * N + mj] > 0)
        M.history[M.history_head] = 3;
    else
        M.history[M.history_head] = 4;
    M.history_head = M.history_head + 1 >= history_size ? 0 : M.history_head + 1;
}

function reset_M(M) {
    M.data.fill(0);
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
    c.width = window.innerWidth - 10;
    c.height = window.innerHeight - 20;
    T++;

    update_M(A, I, K);
    update_M(B, K, J);
    A.data[I * N + K] = L3;
    B.data[K * N + J] = L3;
    C.data[I * N + J]++;

    var offset = (size + padding) * N / 2;
    ctx.font = "48px courier";
    ctx.fillText("A", c.width / 2 - offset * 2.5 - 48, c.height / 2 + offset * 1.25);
    ctx.fillText("B", c.width / 2 + offset * 0.25 - 48, c.height / 2 - offset * 1.5);
    ctx.fillText("C", c.width / 2 + offset * 0.25 - 48, c.height / 2 + offset * 1.25);
    draw_matrix(c.width / 2 - offset * 2.5, c.height / 2 + offset * 0.25, A, ctx, I, K);
    draw_matrix(c.width / 2 + offset * 0.25, c.height / 2 - offset * 2.5, B, ctx, K, J);
    draw_result_matrix(c.width / 2 + offset * 0.25, c.height / 2 + offset * 0.25, C, ctx);

    draw_history(c.width / 2 - offset * 2.5, c.height / 2 + offset * 2.27, A, ctx);
    draw_history(c.width / 2 + offset * 0.25, c.height / 2 - offset * 0.48, B, ctx);

    if (config == 0)
        update_indices(N, 1);
    if (config == 1)
        update_indices_transposed(N, 1);
    if (config == 2)
        update_indices_L3block(N);
    if (config == 3)
        update_indices_L2block(N);
    if (config == 4)
        update_indices_L1block(N);
    if (config == 5)
        update_indices_OpenBLAS();

    if (I == 0 && J == 0 && K == 0)
        reset();
}

tick();

var T = 0;
var id = -1;
var running = 0;
var config = 0;
var INTERVAL = 66;
var L1s = 0;
var L2s = 0;
var L3s = 0;

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
        running = 1;
        id = setInterval(tick, INTERVAL);
    } else if (name == "p" && running == 1) {
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
    if (name == "1") {
        L1s = L1s > 0 ? 0 : 1;
    }
    if (name == "2") {
        L2s = L2s > 0 ? 0 : 1;
    }
    if (name == "3") {
        L3s = L3s > 0 ? 0 : 1;
    }
    if (name == "b") {
        if (config < 2 || config > 4) config = 2;
        else if (config < 4) config++;
        reset();
    } if (name == "o") {
        config = 5;
        reset();
    }

    if (config == 5) {
        L3 = N * N / 4;
        L2 = N * N / 4;
        L1 = L2 / 4;
    } else {
        L3 = N * N / 4;
        L2 = L3 / 4;
        L1 = L2 / 4;
    }
}, false);