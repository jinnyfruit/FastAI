import gradio as gr
from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

categories = ("Dog","Cat")

learn = load_learner("model.pkl")

def classify_image(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(200,200))
label = gr.outputs.Label()
examples = ['dog.jpeg', 'cat.jpeg', 'jinny.jpeg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()