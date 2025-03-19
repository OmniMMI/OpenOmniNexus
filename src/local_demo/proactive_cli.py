import os, torchvision, transformers, tqdm, time, json, subprocess
# torchvision.set_video_backend('video_reader')


from .inference import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        'ffmpeg',
        '-y',
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    subprocess.run(command, check=True)

def main(liveinfer: LiveInfer):
    if liveinfer.video_file is None:
        src_video_path = 'local_demo/assets/water.mp4'
    else:
        src_video_path = liveinfer.video_file
    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('local_demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    save_history_path = src_video_path.replace('.mp4', '.json')
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    liveinfer.load_videos(ffmpeg_video_path)
    liveinfer.input_query_stream('Please inform me when there is a mixter in the video.', video_time=0.0)

    timecosts = []
    pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
    for i in range(liveinfer.num_video_frames): # 1 fps
        # liveinfer.frame_token_interval_threshold -= 0.00175 # decay
        start_time = time.time()
        time.sleep(1/liveinfer.frame_fps) # temporal simulation
        liveinfer.input_video_stream(i / liveinfer.frame_fps)
        query, response = liveinfer()
        end_time = time.time()
        timecosts.append(end_time - start_time)
        fps = (i + 1) / sum(timecosts)
        if query:
            history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(response)
        if not query and not response:
            history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
        pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
        pbar.update(1)
    json.dump(history, open(save_history_path, 'w'), indent=4)
    # print(f'The conversation history has been saved to {save_history_path}.')

if __name__ == '__main__':
    liveinfer = LiveInfer()
    main(liveinfer)