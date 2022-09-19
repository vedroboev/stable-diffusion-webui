import math
import os
import sys
import traceback

import modules.scripts as scripts
import modules.shared as shared
import modules.images as images
import modules.devices as devices
import gradio as gr

from PIL import Image
from modules.ui import gr_show
from modules.shared import state
from modules.processing import Processed, StableDiffusionProcessing, process_images, fix_seed


def get_grid_size(new_width, new_height, upscale_factor=2.0, width=512, height=512, overlap=64):
    non_overlap_width = width - overlap
    non_overlap_height = height - overlap

    cols = math.ceil((new_width * upscale_factor - overlap) / non_overlap_width)
    rows = math.ceil((new_height * upscale_factor - overlap) / non_overlap_height)

    return rows * cols

def calculate_total_steps(images, upscale_factor, patch_width=512, patch_height=512, overlap=64):
    total = 0
    for path in images:
        try:
            img = Image.open(path)
            total += get_grid_size(img.width, img.height, upscale_factor, patch_width, patch_height, overlap)
        except:
            continue
    return total


def sd_upscale(p : StableDiffusionProcessing, img: Image, upscale_overlap: int):
    initial_info = None

    fix_seed(p)
    seed = p.seed

    devices.torch_gc()

    grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=upscale_overlap)
    
    batch_size = p.batch_size

    work = []

    for y, h, row in grid.tiles:
        for tiledata in row:
            work.append(tiledata[2])

    batch_count = math.ceil(len(work) / batch_size)

    print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {batch_count} batches.")

    result_images = []
    for n in range(p.n_iter):
        start_seed = seed + n
        p.seed = start_seed

        work_results = []
        for i in range(batch_count):
            p.batch_size = batch_size
            p.init_images = work[i*batch_size:(i+1)*batch_size]

            state.job = f"Batch {i + 1 + n * batch_count} out of {batch_count}"
            processed = process_images(p)

            if initial_info is None:
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images

        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = images.combine_grid(grid)
    
    return combined_image


class Script(scripts.Script):
    def title(self):
        return "Batch upscaling"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        warning = gr.Label("INFO: Use \"Redraw whole image\" mode for the script to work.")

        input_dir = gr.Textbox(label="Input directory", lines=1)
        output_dir = gr.Textbox(label="Output directory", lines=1)

        with gr.Group():
            upscaler_name = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", visible=False)
            upscaler_scale = gr.Slider(minimum=1.5, maximum=8.0, step=0.25, label="Resize", value=2.0)

        with gr.Group():
            sd_upscale = gr.Checkbox(label="SD upscale", value=True)
            sd_upscale_overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64)

        sd_upscale.change(lambda value: {sd_upscale_overlap : gr_show(value)}, inputs=[sd_upscale], outputs=[sd_upscale_overlap])

        return [warning, input_dir, output_dir, upscaler_name, upscaler_scale, sd_upscale, sd_upscale_overlap]

    def run(self, p, label, input_dir, output_dir, upscaler_index, upscaler_scale, do_sd_upscale, sd_upscale_overlap):
        upscaler = shared.sd_upscalers[upscaler_index]

        images = [file for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)] if os.path.isfile(file)]

        print(f"Will upscale {len(images)} images by {upscaler_scale}x.")

        p.n_iter = 1
        p.batch_size = 1
        p.batch_count = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        state.job_count = calculate_total_steps(images, upscaler_scale, p.height, p.width, sd_upscale_overlap) if do_sd_upscale else len(images)

        for index, path in enumerate(images):
            if state.interrupted:
                break

            try:
                img = Image.open(path)
            except:
                print(f"Error processing {path}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

            state.job = f"Processing image {index + 1} out of {len(images)}:"
            print(state.job)

            upscaled_img = upscaler.upscale(img, img.width * upscaler_scale, img.height * upscaler_scale)
            
            if do_sd_upscale:
                upscaled_img = sd_upscale(p, upscaled_img, sd_upscale_overlap)

            filename = os.path.basename(path)
            upscaled_img.save(os.path.join(output_dir, filename))
            
            if not do_sd_upscale:
                state.nextjob()

        return Processed(p, [], p.seed, "")
