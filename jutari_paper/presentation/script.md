# Speaker Script

A ~5-minute narrated walkthrough of the paper and supplement, in the first-person
plural, matching the paper's voice. Each block is one timeline segment, keyed as
`## Slide N` (N is the timeline order, not the beamer page). Segments 5, 6, and 10
are gameplay/animation clips and are written as a guided walkthrough of what is on
screen; their narration plays continuously over the moving footage. Segments 8 and
9 each paraphrase a theorem in plain words.

**TTS authoring rules (Chatterbox).** No LaTeX is read out. Every symbol is
paraphrased in English, because the synthesiser cannot read math. Sentences stay
under roughly twenty words. No em-dashes, semicolons, or colons in the spoken text
(they trigger stutter). `tts_chatterbox.py` synthesises one sentence at a time, so
clean periods give clean audio.

## Slide 1 — Title (15 s)

We present a differentiable Atari Video Computer System. It is a complex, fully known ground truth for explainable AI. We take a real computer, and we make its every step differentiable. This short video walks through the paper and the supplement.

## Slide 2 — Motivation (35 s)

Explanation needs ground truth. To check that an account of a system is correct, we must know how the system truly works. Today the systems we can study fall into two camps. Some are simple and procedural. Their mechanism is known, but explaining it tests nothing. Others are genuinely complex deep networks. There an explanation is needed, but no inner ground truth exists. So an explanation can be plausible, confident, and wrong. We set out to remove this dichotomy. We build a system that is complex, fully specified, and fully differentiable.

## Slide 3 — The Atari VCS (30 s)

Our study object is the Atari twenty six hundred. It is a real computer from nineteen seventy seven. It has a processor, memory, a graphics chip, and a game cartridge. It is also the platform on which deep reinforcement learning was first established. The machine is complex, yet specified to the last bit. We re-express its every step so that gradients can flow through it.

## Slide 4 — Two ports (30 s)

We built the emulator twice, in two languages. One port is written in Julia, and uses the Zygote gradient system. The other is written in JAX. We call them jutari and jaxtari. Both reproduce the reference emulator exactly. On all sixty four supported games, the memory is byte identical. The screen is pixel identical. Two independent ports, built against one reference, agree to the bit.

## Slide 5 — Conformance, one game (18 s)

Here is one game, side by side. On the left is the reference emulator. In the middle is our port. On the right is the difference between them. Watch the difference panel. It stays solid black. Every pixel matches, on every frame.

## Slide 6 — Conformance, both ports (32 s)

Now we show both ports at once. The top row is the Julia port. The bottom row is the JAX port. In each row, the left is the reference, the middle is our port, and the right is the difference. Watch both difference panels as the games play. They stay black across every game. Space Invaders, then Seaquest, then Enduro, all match to the pixel. Two independent ports, both bit exact, side by side.

## Slide 7 — Soft equals hard (35 s)

How can a bit exact emulator also be differentiable? We re-express each step. The cartridge becomes a weight tensor. The memory becomes a soft tape. Every branch becomes a smooth gate. We prove that the soft forward pass equals the hard machine, bit for bit, at any finite temperature. The backward pass then gives surrogate gradients, exactly where the bit logic has none. The frame stays exact. The gradient appears for free.

## Slide 8 — Theorem one (30 s)

Our first theorem is exact forward equivalence. At any finite sharpness and any finite temperature, the soft machine and the hard machine reach the same state. They agree in the very same thirty two bit numbers. So the soft run matches the hard run at every step. The only difference is the gradient. The soft machine has a useful gradient exactly where the hard one is flat or undefined. This matters because attribution runs on the true execution, not an approximation. The forward error is zero, at any temperature.

## Slide 9 — Theorem two (28 s)

Our second theorem is a temperature limit bound. Here we drop the straight through trick and use the fully relaxed machine. As the temperature falls toward zero, and the sharpness grows, the relaxed step approaches the hard step. The gap shrinks exponentially in both limits. This matters because the relaxation is principled on its own. And the straight through estimator reaches that same hard result exactly, already at a finite temperature.

## Slide 10 — The relaxation, animated (25 s)

These animations show the relaxation at work. On the left, the sharpness parameter grows. The soft gate sharpens toward a hard branch. On the right, the temperature falls. The soft read narrows onto a single memory cell. As both limits tighten, the soft path approaches the exact machine, while the gradients stay well defined.

## Slide 11 — Throughput (25 s)

The JAX port also opens a path to the graphics card. The Julia port is fastest for a single environment on the processor. The JAX port batches many environments together. On a single commodity graphics card, it reaches millions of environment steps per second. The reverse mode gradient costs only a few percent more.

## Slide 12 — Gradients with a ground truth (30 s)

Finally, a proof of concept. We run a real Space Invaders cartridge inside the differentiable substrate. We then take the gradient of the screen with respect to the joystick action. The result highlights exactly the parts of the screen that the action moves. Because the machine is fully known, this attribution can be scored against the truth.

## Slide 13 — In the supplement (25 s)

The supplement holds the details. It gives the full proofs of our two theorems. It studies how the relaxation parameters shape the gradient. It lists every game, with its cartridge hash, its mapper, and its exactness. It reports throughput on the processor and the graphics card. And it includes these comparison videos.

## Slide 14 — Conclusion (22 s)

To summarize. We deliver a complex system that is fully known and fully differentiable. We built it twice, and validated it bit for bit on all sixty four games. It emits exact frames, with surrogate gradients for attribution. We offer it as a ground truth testbed for explainable AI. The full code of both ports will be released under the MIT license upon acceptance. Thank you for your attention.
