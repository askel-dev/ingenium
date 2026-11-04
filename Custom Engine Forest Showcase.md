# Custom Engine Forest Showcase

- **Channel:** Toni Macaroni
- **Published:** 2026-11-03
- **Duration:** 2:45
- **Source:** https://www.youtube.com/watch?v=uR5EvBWxN14

## Description

A new forest showcase in the engine I've been creating for the past 8 months or so (pretty much all of it handwritten).

I have been working on a lot of stuff in Godot since the last video (GI, virtual textures and other things), but I ultimately decided it would make more sense to roll a custom engine with an architecture defined by all of the things I need and not bolted onto some engine that tries to do everything. The engine doesn't do a lot of different things, but what it does (rendering a heavy amount of instances with a heavy amount of polygons) it does incredibly well.

### Architecture

- CLODs (virtual geometry) with voxels for far representations (no impostors are used here).
- Screenspace + Voxel hybrid GI (no raytracing).
- Proxies without instance data for very far away representations.

Additionally I wanted to support expressive weather systems from the ground up (since it's going to be a big part of my game). I'm not 100% there yet but I will keep improving the realism.

If you have questions feel free to ask.

## Screenshots

One frame every 10 seconds of the video, in `docs/showcase/`.

![frame_01](<docs/showcase/frame_01.jpg>)
![frame_02](<docs/showcase/frame_02.jpg>)
![frame_03](<docs/showcase/frame_03.jpg>)
![frame_04](<docs/showcase/frame_04.jpg>)
![frame_05](<docs/showcase/frame_05.jpg>)
![frame_06](<docs/showcase/frame_06.jpg>)
![frame_07](<docs/showcase/frame_07.jpg>)
![frame_08](<docs/showcase/frame_08.jpg>)
![frame_09](<docs/showcase/frame_09.jpg>)
![frame_10](<docs/showcase/frame_10.jpg>)
![frame_11](<docs/showcase/frame_11.jpg>)
![frame_12](<docs/showcase/frame_12.jpg>)
![frame_13](<docs/showcase/frame_13.jpg>)
![frame_14](<docs/showcase/frame_14.jpg>)
![frame_15](<docs/showcase/frame_15.jpg>)
![frame_16](<docs/showcase/frame_16.jpg>)
![frame_17](<docs/showcase/frame_17.jpg>)
