"""The slot for the assistant under test. Copy this file, fill in respond(), and run:

  python packing/generate.py OUT --seed 3 --dry-run --assistant packing/assistant_template.py
  python packing/episodes.py 3 packing/assistant_template.py

As it stands it is empty: it never speaks, so the episode is the user on their own (the
same as --assistant none).

respond(view) is called before every action of the user and once at the end. `view`
(usersim.public_view) holds only what an assistant may see:
  task           {"title", "pack": [item labels]}
  preferences    [{"item", "spot"}] the user stated at the start
  conversation   [{"t", "speaker": "user" | "assistant", "text"}] so far
  now            {"t", "step", "position": [x, y], "room", "holding", "conversation",
                  and after the first action "t_start", "t_end" (+ "frames" in physics)}
  history        the earlier `now`s, oldest first
Return a sentence to say, a list of sentences, or None to stay quiet.
"""


def respond(view):
    return None
