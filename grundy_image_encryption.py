"""
Key-dependent Grundy colouring of random graphs for RGB image encryption.

Reference implementation of the scheme described in the paper.
Encryption: graph construction -> Grundy (first-fit) colouring computed with
the Jones-Plassmann method -> colour-based permutation -> forward and
backward colour-class diffusion, repeated for ROUNDS rounds.
All key material is generated with SHA-256 in counter mode.
"""
import os
import sys
import time
import hashlib
import gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================================
# 0. PARAMETERS
# ==========================================================
ROUNDS = 3          # number of rounds
LONG_EDGES = 2      # long-range edges created per vertex and per round

# 256-bit secret key and 128-bit nonce (demo values).
KEY = bytes.fromhex("00112233445566778899aabbccddeeff"
                    "ffeeddccbbaa99887766554433221100")
NONCE = bytes.fromhex("0f1e2d3c4b5a69788796a5b4c3d2e1f0")


# ==========================================================
# 1. SHA-256 COUNTER-MODE STREAM
# ==========================================================
def sha_stream(key, nonce, label, rnd, nbytes):
    prefix = hashlib.sha256()
    prefix.update(key + b"|" + nonce + b"|" + label.encode("ascii") + b"|" + rnd.to_bytes(2, "big"))
    output = bytearray(nbytes)
    for i, start in enumerate(range(0, nbytes, 32)):
        h = prefix.copy()
        h.update(i.to_bytes(8, "big"))
        output[start:start + min(32, nbytes - start)] = h.digest()[:min(32, nbytes - start)]
    return np.frombuffer(output, dtype=np.uint8)



def stream_u32(key, nonce, label, rnd, count):
    return sha_stream(key, nonce, label, rnd, 4 * count).view("<u4")


# ==========================================================
# 2. GRAPH CONSTRUCTION
# ==========================================================
_STATIC_EDGES = {}

def static_edges(M, N):
    if (M, N) in _STATIC_EDGES:
        return _STATIC_EDGES[(M, N)]
    idx = np.arange(M * N * 3, dtype=np.int32).reshape(M, N, 3)
    pairs = [
        (idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()),      # right neighbour
        (idx[:-1, :, :].ravel(), idx[1:, :, :].ravel()),      # lower neighbour
        (idx[:, :, 0].ravel(), idx[:, :, 1].ravel()),         # R-G
        (idx[:, :, 1].ravel(), idx[:, :, 2].ravel()),         # G-B
        (idx[:, :, 0].ravel(), idx[:, :, 2].ravel()),         # R-B
    ]
    u = np.concatenate([p[0] for p in pairs])
    v = np.concatenate([p[1] for p in pairs])
    _STATIC_EDGES[(M, N)] = (u, v)
    return u, v


def build_graph(M, N, key, nonce, rnd):
    """Same simple undirected graph, without np.unique's large hash table."""
    L = M * N * 3
    if L <= 0 or L > np.iinfo(np.int32).max:
        raise ValueError("Image is empty or too large for 32-bit vertex indices.")
    if LONG_EDGES < 0:
        raise ValueError("LONG_EDGES must be nonnegative.")
    su, sv = static_edges(M, N)
    raw = stream_u32(key, nonce, "edge", rnd, LONG_EDGES * L)
    keys = np.empty(2 * (su.size + raw.size), dtype=np.uint64)
    pos = 0
    chunk = 131072

    def append_pairs(u, v, pos):
        n = u.size
        # Cast before multiplication: int32 would overflow here.
        keys[pos:pos+n] = u.astype(np.uint64) * L + v
        keys[pos+n:pos+2*n] = v.astype(np.uint64) * L + u
        return pos + 2*n

    for start in range(0, su.size, chunk):
        pos = append_pairs(su[start:start+chunk], sv[start:start+chunk], pos)
    if LONG_EDGES:
        for start in range(0, raw.size, chunk):
            stop = min(start + chunk, raw.size)
            u = np.arange(start, stop, dtype=np.int64) // LONG_EDGES
            v = (raw[start:stop] % L).astype(np.int32)
            keep = u != v
            pos = append_pairs(u[keep], v[keep], pos)
    del raw
    keys = keys[:pos]
    keys.sort(kind="quicksort")  # in-place sort, no hash table
    keep = np.empty(keys.size, dtype=bool)
    keep[0] = True
    keep[1:] = keys[1:] != keys[:-1]
    # Compact in bounded chunks; output never overwrites unread input.
    n = 0
    for start in range(0, keys.size, chunk):
        selected = keys[start:start+chunk][keep[start:start+chunk]]
        keys[n:n+selected.size] = selected
        n += selected.size
    del keep
    src = np.empty(n, dtype=np.int32)
    dst = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        block = keys[start:min(start+chunk, n)]
        src[start:start+block.size] = block // L
        dst[start:start+block.size] = block % L
    return src, dst


# ==========================================================
# 3. GRUNDY (FIRST-FIT) COLOURING  -- parallel Jones-Plassmann
# ==========================================================
def grundy_coloring(L, src, dst, rank):
    color = np.full(L, -1, dtype=np.int32)
    uncolored = np.ones(L, dtype=bool)
    kmax = int(np.bincount(src, minlength=L).max()) + 2

    act_src, act_dst = src, dst                      
    lut = np.zeros(L, dtype=np.int32)

    while act_src.size > 0 or uncolored.any():
        blocked_edges = uncolored[act_dst] & (rank[act_dst] < rank[act_src])
        blocked = np.zeros(L, dtype=bool)
        blocked[act_src[blocked_edges]] = True
        sel = uncolored & ~blocked
        sel_idx = np.nonzero(sel)[0]

        em = sel[act_src]
        es, ed = act_src[em], act_dst[em]
        cd = color[ed]
        valid = cd >= 0
        lut[sel_idx] = np.arange(sel_idx.size)
        present = np.zeros((sel_idx.size, kmax), dtype=bool)
        present[lut[es[valid]], cd[valid]] = True
        color[sel_idx] = np.argmin(present, axis=1)

        uncolored[sel_idx] = False
        keep = uncolored[act_src]
        act_src, act_dst = act_src[keep], act_dst[keep]
        if not uncolored.any():
            break
    return color


def check_coloring(color, src, dst):
    L = color.size
    k = int(color.max()) + 1
    proper = not np.any(color[src] == color[dst])
    present = np.zeros((L, k), dtype=bool)
    present[src, color[dst]] = True
    need = np.arange(k)[None, :] < color[:, None]
    grundy = bool(np.all(present | ~need))
    return proper, grundy, k


# ==========================================================
# 4. ONE ROUND: ALL KEY-DEPENDENT DATA
# ==========================================================
def build_round(M, N, key, nonce, rnd):
    L = M * N * 3
    src, dst = build_graph(M, N, key, nonce, rnd)

    order_keys = stream_u32(key, nonce, "order", rnd, L)
    order = np.argsort(order_keys, kind="stable")
    rank = np.empty(L, dtype=np.int32)
    rank[order] = np.arange(L, dtype=np.int32)
    del order_keys, order
    color = grundy_coloring(L, src, dst, rank)
    del rank
    k = int(color.max()) + 1
    if k >= 256:
        raise ValueError("The rotation table supports fewer than 256 colours.")
    color = color.astype(np.uint8)
    perm_keys = stream_u32(key, nonce, "perm", rnd, L)
    T = np.lexsort((perm_keys, color)).astype(np.int32)
    del perm_keys
    members = [np.flatnonzero(color == j).astype(np.int32) for j in range(k)]
    local = np.empty(L, dtype=np.int32)
    for verts in members:
        local[verts] = np.arange(verts.size, dtype=np.int32)

    # Store each directed edge index once, with compact 32-bit indices.
    cs, cd = color[src], color[dst]
    lower_split, upper_split = [], []
    for j in range(k):
        lower_split.append(np.flatnonzero((cs == j) & (cd < j)).astype(np.int32))
        upper_split.append(np.flatnonzero((cs == j) & (cd > j)).astype(np.int32))
    del cs, cd
    rho = sha_stream(key, nonce, "rho", rnd, 512)
    return dict(L=L, k=k, src=src, dst=dst, color=color, T=T,
                members=members, local=local,
                lower_split=lower_split, upper_split=upper_split,
                r_f=sha_stream(key, nonce, "rf", rnd, L),
                m_f=sha_stream(key, nonce, "mf", rnd, L),
                r_b=sha_stream(key, nonce, "rb", rnd, L),
                m_b=sha_stream(key, nonce, "mb", rnd, L),
                rho_f=1 + rho[:256] % 7, rho_b=1 + rho[256:] % 7)


def rotl(t, r):
    return ((t << r) | (t >> (8 - r))) & 255


def rotr(t, r):
    return ((t >> r) | (t << (8 - r))) & 255


# ==========================================================
# 5. ENCRYPTION / DECRYPTION OF ONE ROUND
# ==========================================================
def encrypt_round(x, rd):
    L, k = rd["L"], rd["k"]
    src, dst, local, members = rd["src"], rd["dst"], rd["local"], rd["members"]

    s = x[rd["T"]]

    y = np.zeros(L, dtype=np.uint8)
    for j in range(k):
        verts = members[j]
        e = rd["lower_split"][j]
        sums = np.bincount(local[src[e]], weights=y[dst[e]], minlength=verts.size).astype(np.int64)
        t = (s[verts].astype(np.int64) + sums + rd["r_f"][verts]) & 255
        y[verts] = ((rotl(t, int(rd["rho_f"][j])) ^ rd["m_f"][verts]) & 255).astype(np.uint8)

    z = np.zeros(L, dtype=np.uint8)
    for j in range(k - 1, -1, -1):
        verts = members[j]
        e = rd["upper_split"][j]
        sums = np.bincount(local[src[e]], weights=z[dst[e]], minlength=verts.size).astype(np.int64)
        t = (y[verts].astype(np.int64) + sums + rd["r_b"][verts]) & 255
        z[verts] = ((rotl(t, int(rd["rho_b"][j])) ^ rd["m_b"][verts]) & 255).astype(np.uint8)
    return z


def decrypt_round(z, rd):
    L = rd["L"]
    src, dst, local = rd["src"], rd["dst"], rd["local"]
    y, s = np.empty(L, dtype=np.uint8), np.empty(L, dtype=np.uint8)
    # Undo backward diffusion using the complete ciphertext z.
    for j, verts in enumerate(rd["members"]):
        e = rd["upper_split"][j]
        sums = np.bincount(local[src[e]], weights=z[dst[e]], minlength=verts.size).astype(np.int64)
        v = z[verts].astype(np.int64) ^ rd["m_b"][verts]
        y[verts] = (rotr(v, int(rd["rho_b"][j])) - sums - rd["r_b"][verts]) & 255
    # Undo forward diffusion using the complete recovered y.
    for j, verts in enumerate(rd["members"]):
        e = rd["lower_split"][j]
        sums = np.bincount(local[src[e]], weights=y[dst[e]], minlength=verts.size).astype(np.int64)
        v = y[verts].astype(np.int64) ^ rd["m_f"][verts]
        s[verts] = (rotr(v, int(rd["rho_f"][j])) - sums - rd["r_f"][verts]) & 255
    x = np.empty(L, dtype=np.uint8)
    x[rd["T"]] = s
    return x


def validate_image(image, rounds):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Use a NumPy uint8 RGB image.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Expected a nonempty image of shape (height, width, 3).")
    if not isinstance(rounds, int) or not 1 <= rounds <= 65536:
        raise ValueError("rounds must be an integer from 1 to 65536.")


# ==========================================================
# 6. FULL IMAGE
# ==========================================================
def encrypt_image(image, key=KEY, nonce=NONCE, rounds=ROUNDS, verbose=False):
    validate_image(image, rounds)
    M, N, _ = image.shape
    x = image.reshape(-1).copy()
    for rnd in range(rounds):
        t0 = time.time()
        rd = build_round(M, N, key, nonce, rnd)
        x = encrypt_round(x, rd)
        if verbose:
            print(f"  round {rnd + 1}/{rounds}: {rd['k']} colours, {time.time() - t0:.1f} s")
        del rd
        gc.collect()
    return x.reshape(M, N, 3)


def decrypt_image(cipher, key=KEY, nonce=NONCE, rounds=ROUNDS, verbose=False):
    validate_image(cipher, rounds)
    M, N, _ = cipher.shape
    x = cipher.reshape(-1).copy()
    for rnd in reversed(range(rounds)):
        t0 = time.time()
        rd = build_round(M, N, key, nonce, rnd)
        x = decrypt_round(x, rd)
        if verbose:
            print(f"  round {rnd + 1}: decrypted, {time.time() - t0:.1f} s")
        del rd
        gc.collect()
    return x.reshape(M, N, 3)


# ==========================================================
# 7. METRICS
# ==========================================================
def calc_npcr_uaci_rgb(img1, img2):
    out = []
    for c in range(3):
        a = img1[:, :, c].ravel().astype(np.float64)
        b = img2[:, :, c].ravel().astype(np.float64)
        out.append((np.mean(a != b) * 100.0, np.mean(np.abs(a - b)) / 255.0 * 100.0))
    return out


def calc_entropy(image):
    res = []
    for c in range(3):
        _, counts = np.unique(image[:, :, c].ravel(), return_counts=True)
        p = counts / counts.sum()
        res.append(float(-np.sum(p * np.log2(p))))
    return res


def get_adjacent_pairs(img, channel_idx, direction="h", samples=3000):
    ch = img[:, :, channel_idx]
    if direction == "h":
        x, y = ch[:, :-1].ravel(), ch[:, 1:].ravel()
    elif direction == "v":
        x, y = ch[:-1, :].ravel(), ch[1:, :].ravel()
    else:
        x, y = ch[:-1, :-1].ravel(), ch[1:, 1:].ravel()
    idx = np.random.default_rng(12345).choice(len(x), size=min(samples, len(x)), replace=False)
    return x[idx], y[idx]


def correlation_coefficient(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.size == 0:
        return float("nan")
    sx, sy = x.std(), y.std()
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)) if sx * sy > 0 else 0.0


def calc_mse_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return mse, (float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))


# ==========================================================
# 8. 3D CORRELATION FIGURE
# ==========================================================
def plot_correlation_3d(plain_image, cipher_image, samples=3000, save_path="correlation_3d.png"):
    dirs = [("h", "Horizontal", r"$(x,y)$", r"$(x+1,y)$"),
            ("v", "Vertical", r"$(x,y)$", r"$(x,y+1)$"),
            ("d", "Diagonal", r"$(x,y)$", r"$(x+1,y+1)$")]
    ch_colors = ["red", "lime", "blue"]
    depth = [0, 1, 2]
    letters = "abcdef"
    y_lo, y_hi = -0.2, 2.2
    fig = plt.figure(figsize=(18, 11))
    for row, img in enumerate((plain_image, cipher_image)):
        for col, (d, name, xl, zl) in enumerate(dirs):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            for c_idx in range(3):
                xs, ys = get_adjacent_pairs(img, c_idx, d, samples)
                ax.scatter(xs, np.full(len(xs), float(depth[c_idx])), ys, s=3,
                           color=ch_colors[c_idx], depthshade=False, linewidths=0)
            if row == 0:
                for z in (0, 255):
                    for yy in (y_lo, y_hi):
                        ax.plot([0, 255], [yy, yy], [z, z], color="k", lw=0.8)
                    for xx in (0, 255):
                        ax.plot([xx, xx], [y_lo, y_hi], [z, z], color="k", lw=0.8)
                for xx in (0, 255):
                    for yy in (y_lo, y_hi):
                        ax.plot([xx, xx], [yy, yy], [0, 255], color="k", lw=0.8)
            else:
                for c_idx in range(3):
                    ax.plot([0, 255, 255, 0, 0], [depth[c_idx]] * 5, [0, 0, 255, 255, 0],
                            color="k", lw=0.8)
            ax.set_xlim(0, 255); ax.set_ylim(y_lo, y_hi); ax.set_zlim(0, 255)
            ax.set_xticks(range(0, 251, 50)); ax.set_zticks(range(0, 251, 50)); ax.set_yticks([])
            ax.set_xlabel(xl, fontsize=11); ax.set_zlabel(zl, fontsize=11)
            ax.tick_params(labelsize=9)
            ax.view_init(elev=20, azim=-55)
            ax.set_box_aspect((1.1, 1.0, 0.9))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((1, 1, 1, 0))
                axis.pane.set_edgecolor((1, 1, 1, 0))
                axis._axinfo["grid"]["linewidth"] = 0
            ax.text2D(0.5, -0.02, f"({letters[row * 3 + col]}) {name}", transform=ax.transAxes,
                      ha="center", va="top", fontsize=16)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06, wspace=0.0, hspace=0.12)
    fig.savefig(save_path, dpi=300)
    return fig


# ==========================================================
# 9. MAIN
# ==========================================================
if __name__ == "__main__":
    # Usage: python grundy_image_encryption.py [path/to/image.png]
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("images", "baboon.png")

    if os.path.isfile(img_path):
        plain = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    else:
        print(f"File not found: {img_path}\nUsing a synthetic 256x256 test image instead.")
        rng = np.random.default_rng(1)
        base = np.cumsum(np.cumsum(rng.normal(size=(256, 256)), 0), 1)
        base = (base - base.min()) / (base.max() - base.min())
        plain = np.stack([np.clip(base * 255 * s + rng.normal(0, 6, base.shape), 0, 255)
                          for s in (1.0, 0.9, 0.8)], axis=2).astype(np.uint8)

    M, N, _ = plain.shape
    print(f"Image: {M} x {N} x 3, rounds = {ROUNDS}, long-range edges per vertex = {LONG_EDGES}")

    # ---- graph colouring check (round 0) ----
    print("\n[ 0. Graph colouring (round 1) ]")
    rd0 = build_round(M, N, KEY, NONCE, 0)
    proper, grundy, k = check_coloring(rd0["color"], rd0["src"], rd0["dst"])
    print(f"Vertices: {rd0['L']}, directed edges: {rd0['src'].size}")
    print(f"Number of colours in this Grundy colouring: {k}")
    print(f"Proper colouring: {proper}, Grundy property: {grundy}")
    print("Class sizes:", [m.size for m in rd0["members"]])
    
    del rd0
    gc.collect()

    # ---- encryption / decryption ----
    print("\n--- ENCRYPTION ---")
    t0 = time.time()
    cipher = encrypt_image(plain, verbose=True)
    enc_time = time.time() - t0
    print(f"Encryption time: {enc_time:.2f} s")

    print("\n--- DECRYPTION ---")
    t0 = time.time()
    decrypted = decrypt_image(cipher, verbose=True)
    dec_time = time.time() - t0
    print(f"Decryption time: {dec_time:.2f} s")
    exact = np.array_equal(plain, decrypted)
    print("Exact recovery:", exact)
    if not exact:
        raise RuntimeError("Decryption verification failed.")
    Image.fromarray(cipher).save("encrypted.png")
    Image.fromarray(decrypted).save("decrypted.png")

    print("\n[ 1. Image quality (plain vs cipher) ]")
    mse, psnr = calc_mse_psnr(plain, cipher)
    print(f"MSE: {mse:.4f}, PSNR: {psnr:.4f} dB")

    labels = ["Red", "Green", "Blue"]
    gc.collect()

    # ---- key sensitivity: one bit of the key ----
    print("\n[ 2. Key sensitivity: one bit of the key changed ]")
    key2 = bytearray(KEY)
    key2[-1] ^= 1
    cipher_k = encrypt_image(plain, key=bytes(key2))
    res = calc_npcr_uaci_rgb(cipher, cipher_k)
    for i, (npcr, uaci) in enumerate(res):
        print(f"Channel {labels[i]} -> NPCR: {npcr:.4f}%, UACI: {uaci:.4f}%")
    print(f"Average         -> NPCR: {np.mean([r[0] for r in res]):.4f}%, "
          f"UACI: {np.mean([r[1] for r in res]):.4f}%")
          
    gc.collect() 
    
    wrong = decrypt_image(cipher, key=bytes(key2))
    print("Decryption with the wrong key recovers the image:", np.array_equal(plain, wrong))
    
    del cipher_k, wrong
    gc.collect()

    # ---- plaintext sensitivity: one bit of the first red pixel ----
    print("\n[ 3. Plaintext sensitivity: one bit of pixel (0,0,R) changed ]")
    plain2 = plain.copy()
    plain2[0, 0, 0] ^= 1
    cipher_p = encrypt_image(plain2)
    res = calc_npcr_uaci_rgb(cipher, cipher_p)
    for i, (npcr, uaci) in enumerate(res):
        print(f"Channel {labels[i]} -> NPCR: {npcr:.4f}%, UACI: {uaci:.4f}%")
    print(f"Average         -> NPCR: {np.mean([r[0] for r in res]):.4f}%, "
          f"UACI: {np.mean([r[1] for r in res]):.4f}%")
          
    del plain2, cipher_p
    gc.collect()

    # ---- correlation ----
    print("\n[ 4. Correlation coefficients ]")
    for d, dl in zip(["h", "v", "d"], ["Horizontal", "Vertical", "Diagonal"]):
        print(f"--- {dl} ---")
        for c in range(3):
            xp, yp = get_adjacent_pairs(plain, c, d)
            xc, yc = get_adjacent_pairs(cipher, c, d)
            print(f"{labels[c]}: Plain = {correlation_coefficient(xp, yp):.4f} | "
                  f"Cipher = {correlation_coefficient(xc, yc):.4f}")

    # ---- entropy ----
    print("\n[ 5. Information entropy ]")
    ep, ec = calc_entropy(plain), calc_entropy(cipher)
    for c in range(3):
        print(f"Channel {labels[c]} -> Plain: {ep[c]:.4f} | Cipher: {ec[c]:.4f}")

    # ---- figures ----
    fig1, ax1 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, tt in zip(ax1, (plain, cipher, decrypted), ("Plain Image", "Encrypted Image", "Decrypted Image")):
        ax.imshow(im); ax.set_title(tt); ax.axis("off")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Histogram Analysis", fontsize=16)
    for ax, im, tt in zip(ax2, (plain, cipher, decrypted), ("Plain Image", "Cipher Image", "Decrypted Image")):
        for i, col in enumerate(["red", "green", "blue"]):
            ax.hist(im[:, :, i].ravel(), bins=256, range=(0, 256), color=col, alpha=0.5,
                    label=f"{col.capitalize()} Channel")
        ax.set_title(tt); ax.set_xlabel("Pixel Intensity"); ax.set_ylabel("Frequency")
        ax.set_xlim([0, 256]); ax.legend()
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    plot_correlation_3d(plain, cipher, samples=3000)
    plt.show()