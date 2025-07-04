"""
This class maintains the state of the animation, including the positions,
names, and widths of the elements being animated.
"""


class AnimationState:
    def __init__(self, top_n):
        self.prev_interp_positions = self.prev_positions = (
            self.current_new_positions
        ) = list(range(9, 9 - top_n, -1))
        self.prev_names = [""] * top_n
        self.prev_widths = [0] * top_n
