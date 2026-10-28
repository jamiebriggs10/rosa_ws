"""The voice front end of the console: the microphone in, the words out, Rosa's voice back.

    words.py     what a spoken phrase means: "stop", "confirm", "no, point eleven", the wake word
    audio.py     the microphone and WAV files as 32 ms frames at 16 kHz
    stages.py    the wake word, push-to-talk, Silero VAD, the two end-of-turn rules, Parakeet
    pipeline.py  the state machine that drives the stages, frame by frame, on its own thread
    speech.py    Piper for Rosa's lines and the loop's short sounds
    models.py    the model files, downloaded once into models/

The `Voice` object at the bottom of this file is what the console owns: it builds the stages,
runs the pipeline on a thread and hands what it hears to the console through callbacks.
It is filled in by Task 5 of the plan; until then this package is words and audio.
"""
