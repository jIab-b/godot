import os
import uuid
import sys
import struct
import json
import subprocess
import tempfile
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

if os.name == "nt":
    _default_trt_root = r"C:\\Program Files\\TensorRT"
    _trt_root = os.environ.get("TENSORRT_ROOT", _default_trt_root)
    _trt_bin_dir = os.path.join(_trt_root, "bin")
    if os.path.isdir(_trt_bin_dir):
        os.environ["PATH"] = _trt_bin_dir + ";" + os.environ.get("PATH", "")

class TensorRTSDXLPipeline:
    def __init__(self):
        pass

    def preprocess_depth(self, depth_image_path):
        img = Image.open(depth_image_path).convert("RGB")
        img = img.resize((1024, 1024))
        arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float16)
        return arr

    def _extract_output_values(self, j, prefer_keys=None):
        def pick(v):
            if isinstance(v, dict):
                if prefer_keys:
                    for k in prefer_keys:
                        if k in v:
                            return pick(v[k])
                if "values" in v:
                    return v["values"]
                if "data" in v:
                    return v["data"]
                if "buffer" in v:
                    return v["buffer"]
                return pick(next(iter(v.values()))) if v else []
            if isinstance(v, list):
                if v and isinstance(v[0], (int, float)):
                    return v
                if v and isinstance(v[0], dict):
                    if "outputs" in v[0]:
                        out0 = v[0]["outputs"][0]
                        return out0.get("values", out0.get("data", out0.get("buffer", [])))
                    return pick(v[0])
            return v

        return pick(j)

    def _encode_prompt_ids_for_tokenizer(self, prompt, model_id, subfolder):
        try:
            from transformers import CLIPTokenizer
        except Exception:
            raise RuntimeError("transformers not available; provide --prompt_ids or install transformers")
        tok = CLIPTokenizer.from_pretrained(model_id, subfolder=subfolder)
        enc = tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np")
        return enc.input_ids.astype("int64").tobytes()

    def _run_trtexec(self, engine_path, shapes, inputs_bin_map, export_json=None, extra_args=None):
        cmd = ["trtexec", f"--loadEngine={engine_path}"]
        if shapes:
            shape_specs = []
            for name, shp in shapes.items():
                shape_specs.append(f"{name}:{shp}")
            cmd.append(f"--shapes={','.join(shape_specs)}")
        if inputs_bin_map:
            load_specs = []
            for name, fpath in inputs_bin_map.items():
                load_specs.append(f"{name}:{fpath}")
            cmd.append(f"--loadInputs={','.join(load_specs)}")
        if export_json:
            cmd.append(f"--exportOutput={export_json}")
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout)
        return proc.stdout

    def _flatten_fp32_to_fp16_bytes(self, floats):
        out = bytearray(2 * len(floats))
        for i, v in enumerate(floats):
            out[2*i:2*i+2] = struct.pack("<e", float(v))
        return bytes(out)

    def _read_int_ids(self, path, length):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        parts = [p for p in txt.replace("\n", ",").split(",") if p.strip()]
        ids = [int(p) for p in parts]
        if len(ids) != length:
            raise ValueError("incorrect token length")
        return b"".join(struct.pack("<q", v) for v in ids)

    def _concat_last_dim_fp16(self, a_bytes, a_shape, b_bytes, b_shape):
        n, t, d1 = a_shape
        n2, t2, d2 = b_shape
        if not (n == n2 and t == t2):
            raise ValueError("shape mismatch")
        out = bytearray(2 * n * t * (d1 + d2))
        row = 2 * (d1 + d2)
        row_a = 2 * d1
        row_b = 2 * d2
        for i in range(n * t):
            start = i * row
            out[start:start + row_a] = a_bytes[i * row_a:(i + 1) * row_a]
            out[start + row_a:start + row_a + row_b] = b_bytes[i * row_b:(i + 1) * row_b]
        return bytes(out)

    def _randn_fp16(self, numel):
        import random
        out = bytearray(2 * numel)
        for i in range(numel):
            v = random.uniform(-1.0, 1.0)
            out[2 * i:2 * i + 2] = struct.pack("<e", v)
        return bytes(out)

    def _axpy_fp16_inplace(self, x_bytes, y_bytes, alpha):
        numel = len(x_bytes) // 2
        out = bytearray(len(x_bytes))
        a = alpha
        for i in range(numel):
            xv = struct.unpack("<e", x_bytes[2 * i:2 * i + 2])[0]
            yv = struct.unpack("<e", y_bytes[2 * i:2 * i + 2])[0]
            out[2 * i:2 * i + 2] = struct.pack("<e", xv - a * yv)
        return bytes(out)

    def _save_ppm_from_chw_fp16(self, chw_bytes, path):
        c3 = 3
        h = 1024
        w = 1024
        data = bytearray(w * h * c3)
        for i in range(h * w):
            r = struct.unpack("<e", chw_bytes[2 * (0 * h * w + i):2 * (0 * h * w + i) + 2])[0]
            g = struct.unpack("<e", chw_bytes[2 * (1 * h * w + i):2 * (1 * h * w + i) + 2])[0]
            b = struct.unpack("<e", chw_bytes[2 * (2 * h * w + i):2 * (2 * h * w + i) + 2])[0]
            rv = max(0, min(255, int((r + 1.0) * 127.5)))
            gv = max(0, min(255, int((g + 1.0) * 127.5)))
            bv = max(0, min(255, int((b + 1.0) * 127.5)))
            data[3 * i:3 * i + 3] = bytes((rv, gv, bv))
        with open(path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(data)

    def encode_prompt_ids(self, ids_path_te1=None, ids_path_te2=None, prompt_text=None):
        uid = uuid.uuid4().hex
        def write_ids_to(path, ids_bytes):
            with open(path, "wb") as f:
                f.write(ids_bytes)
        if prompt_text is not None:
            ids_te1 = self._encode_prompt_ids_for_tokenizer(prompt_text, "stabilityai/stable-diffusion-xl-base-1.0", "tokenizer")
            ids_te2 = self._encode_prompt_ids_for_tokenizer(prompt_text, "stabilityai/stable-diffusion-xl-base-1.0", "tokenizer_2")
            inp1 = f"input_ids_te1_{uid}.bin"
            inp2 = f"input_ids_te2_{uid}.bin"
            write_ids_to(inp1, ids_te1)
            write_ids_to(inp2, ids_te2)
        else:
            if ids_path_te1 is None and ids_path_te2 is None:
                raise ValueError("either prompt_text or at least one ids path must be provided")
            # fallback: if only one provided, reuse for both
            inp1 = f"input_ids_te1_{uid}.bin"
            inp2 = f"input_ids_te2_{uid}.bin"
            write_ids_to(inp1, self._read_int_ids(ids_path_te1 or ids_path_te2, 77))
            write_ids_to(inp2, self._read_int_ids(ids_path_te2 or ids_path_te1, 77))
        out_json1 = f"te1_{uid}.json"
        self._run_trtexec(
            engine_path="text_encoder_1.trt",
            shapes={"input_ids": "1x77"},
            inputs_bin_map={"input_ids": inp1},
            export_json=out_json1
        )
        out_json2 = f"te2_{uid}.json"
        self._run_trtexec(
            engine_path="text_encoder_2.trt",
            shapes={"input_ids": "1x77"},
            inputs_bin_map={"input_ids": inp2},
            export_json=out_json2
        )
        def read_json_vecs(p, prefer_keys=None):
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            return self._extract_output_values(j, prefer_keys)
        v1 = read_json_vecs(out_json1, prefer_keys=["text_embeddings"])
        v2 = read_json_vecs(out_json2, prefer_keys=["text_embeddings"])
        b1 = self._flatten_fp32_to_fp16_bytes(v1)
        b2 = self._flatten_fp32_to_fp16_bytes(v2)
        return self._concat_last_dim_fp16(b1, (1, 77, 768), b2, (1, 77, 1280))

    def run_unet(self, latents_bytes, timestep, text_embeddings_bytes, adapter_feats):
        uid = uuid.uuid4().hex
        lat_path = f"sample_{uid}.bin"
        ts_path = f"timestep_{uid}.bin"
        emb_path = f"encoder_hidden_states_{uid}.bin"
        with open(lat_path, "wb") as f:
            f.write(latents_bytes)
        with open(ts_path, "wb") as f:
            f.write(struct.pack("<e", float(timestep)))
        with open(emb_path, "wb") as f:
            f.write(text_embeddings_bytes)
        # Added zero text_embeds/time_ids
        te_path = f"text_embeds_{uid}.bin"
        ti_path = f"time_ids_{uid}.bin"
        with open(te_path, "wb") as f:
            for _ in range(1280):
                f.write(struct.pack("<e", 0.0))
        with open(ti_path, "wb") as f:
            for _ in range(6):
                f.write(struct.pack("<e", 0.0))
        # Adapter feature tensors
        ad_paths = []
        ad_shapes = [(1,320,128,128),(1,640,64,64),(1,1280,32,32),(1,1280,16,16)]
        for i, (arr, shp) in enumerate(zip(adapter_feats, ad_shapes)):
            p = f"adapter_feat_{i}_{uid}.bin"
            with open(p, "wb") as f:
                f.write(arr)
            ad_paths.append(p)
        out_json = f"unet_{uid}.json"
        self._run_trtexec(
            engine_path="sdxl_unet.trt",
            shapes={"sample": "1x4x128x128", "timestep": "1", "encoder_hidden_states": "1x77x2048", "text_embeds": "1x1280", "time_ids": "1x6", "adapter_feat_0": "1x320x128x128", "adapter_feat_1": "1x640x64x64", "adapter_feat_2": "1x1280x32x32", "adapter_feat_3": "1x1280x16x16"},
            inputs_bin_map={"sample": lat_path, "timestep": ts_path, "encoder_hidden_states": emb_path, "text_embeds": te_path, "time_ids": ti_path, "adapter_feat_0": ad_paths[0], "adapter_feat_1": ad_paths[1], "adapter_feat_2": ad_paths[2], "adapter_feat_3": ad_paths[3]},
            export_json=out_json
        )
        with open(out_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        vals = self._extract_output_values(j, prefer_keys=["images"])
        return self._flatten_fp32_to_fp16_bytes(vals)

    def decode_latents(self, latents_bytes):
        uid = uuid.uuid4().hex
        lat_path = f"vae_input_{uid}.bin"
        with open(lat_path, "wb") as f:
            f.write(latents_bytes)
        out_json = f"vae_{uid}.json"
        self._run_trtexec(
            engine_path="vae_decoder.trt",
            shapes={"latents": "1x4x128x128"},
            inputs_bin_map={"latents": lat_path},
            export_json=out_json
        )
        with open(out_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        vals = self._extract_output_values(j)
        # vals length should be 1*3*1024*1024 floats
        # convert to fp16 bytes in CHW order
        return self._flatten_fp32_to_fp16_bytes(vals)

    def generate(self, prompt_ids_path, steps, output_ppm_path, prompt_text=None, prompt_ids_path_te2=None, depth_png_path=None):
        text_embeddings = self.encode_prompt_ids(
            ids_path_te1=prompt_ids_path,
            ids_path_te2=prompt_ids_path_te2,
            prompt_text=prompt_text,
        )
        latents = self._randn_fp16(1 * 4 * 128 * 128)
        adapter_feats = None
        if depth_png_path and os.path.exists(depth_png_path):
            depth = self.preprocess_depth(depth_png_path)
            uid = uuid.uuid4().hex
            in_path = f"depth_{uid}.bin"
            with open(in_path, "wb") as f:
                f.write(depth.tobytes())
            out_json = f"adapter_{uid}.json"
            self._run_trtexec(
                engine_path="t2i_adapter.trt",
                shapes={"depth_input": "1x3x1024x1024"},
                inputs_bin_map={"depth_input": in_path},
                export_json=out_json
            )
            with open(out_json, "r", encoding="utf-8") as f:
                j = json.load(f)
            vals = []
            for i in range(4):
                key = f"res_{i}"
                arr = self._extract_output_values(j, prefer_keys=[key])
                vals.append(self._flatten_fp32_to_fp16_bytes(arr))
            adapter_feats = vals
        for _ in range(steps):
            noise = self.run_unet(latents, 999.0, text_embeddings, adapter_feats if adapter_feats else [self._randn_fp16(1*320*128*128), self._randn_fp16(1*640*64*64), self._randn_fp16(1*1280*32*32), self._randn_fp16(1*1280*16*16)])
            latents = self._axpy_fp16_inplace(latents, noise, 0.5)
        img_chw = self.decode_latents(latents)
        self._save_ppm_from_chw_fp16(img_chw, output_ppm_path)
        return output_ppm_path


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--prompt_ids")
    group.add_argument("--prompt")
    parser.add_argument("--prompt_ids_te2")
    parser.add_argument("--tokenizer", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_ppm", default="tensorrt_output.ppm")
    parser.add_argument("--output_png")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--depth", default="out_depth.png")
    args = parser.parse_args()
    pipeline = TensorRTSDXLPipeline()
    ids_path = None
    ids_path = args.prompt_ids
    out_path = pipeline.generate(
        prompt_ids_path=ids_path,
        prompt_ids_path_te2=args.prompt_ids_te2,
        prompt_text=args.prompt,
        steps=args.steps,
        output_ppm_path=args.output_ppm,
        depth_png_path=args.depth if os.path.exists(args.depth) else None
    )
    print(f"Image saved to {out_path}")
    if args.output_png:
        try:
            from PIL import Image
            Image.open(out_path).save(args.output_png)
            print(f"PNG saved to {args.output_png}")
        except Exception as e:
            print(f"Failed to save PNG: {e}")


if __name__ == "__main__":
    main()
