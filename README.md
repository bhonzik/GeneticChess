The best way that I that can describe this project is as a genetic adversarial algorithm that
gradually learns to get good at playing chess by playing against itself-- there's a model
for the white pieces and a model for the black pieces, and in the beginning, slightly
mutated versions of the model for white play against the model for black until one of the
versions wins, then that version becomes the model for white, and then slightly mutated
versions of the model for black play against the model for white until one of the versions
wins, then that version becomes the model for black, and with that, one epoch has passed.

I'll warn you that even after training for half a day, this model does not perform very well,
even against itself, but since this model requires no data for training, its potential
is theoretically limitless-- a model is only as good as the data that it's trained on,
whether it's a model trained through supervised learning or a genetic algorithm, but this
model learns in a vacuum, so while it improves slowly, there should be no limit to how much
it can improve.

My python skills are rusty and I'm relatively new to machine learning, so I might have no
idea what I'm talking about, but hopefully you'll find this project interesting, or at
least the idea of it.
