import Augmentor
data="C:/Users/YOGAPRIYA/PycharmProjects/Face/database_new/mathew"
pipe=Augmentor.Pipeline(data)
# Add some operations to an existing pipeline.

# First, we add a horizontal flip operation to the pipeline:



pipe.flip_left_right(probability=0.4)

# Now we add a vertical flip operation to the pipeline:
pipe.flip_top_bottom(probability=0.8)

# Add a rotate90 operation to the pipeline:
pipe.rotate270(probability=0.1)
pipe.rotate_random_90(probability=0.2)
pipe.black_and_white(probability=0.1,threshold=128)

pipe.rotate(probability=0.2, max_left_rotation=10, max_right_rotation=10)
pipe.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
number_of_img=int(1000)
pipe.sample(number_of_img)