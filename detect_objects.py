import numpy as np
import pathlib
import sys
import tensorflow as tf
from PIL import Image
from pascal_voc_writer import Writer
import os.path
import ntpath
import re

# sys.path.append("/Users/chadd/Documents/Chadd/Work/DSO/Model_Re-training/TensorFlow/workspace/tf/research")
save_path = "/Users/chadd/Documents/Chadd/Work/DSO/Model_Re-training/TensorFlow/workspace/training/detected_images/upperbound_test"

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
PATH_TO_TEST_IMAGES_DIR = pathlib.Path("images/test")
TEST_IMAGE_PATHS = list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg"))


def main():
    model = load_model()
    ####### initialise a writer to create a pascal voc file #######
    total_count = 0
    for image_path in TEST_IMAGE_PATHS:
        print(image_path)
        count = re.findall(r'[^\/]+(?=\.)', str(image_path))
        writer = Writer(
            '/Users/chadd/Documents/Chadd/Work/DSO/Model_Re-training/TensorFlow/workspace/training/images/test/' +
                count[0] + '.jpg', 256, 256)
        show_inference(model, image_path, writer, count[0])
        print(total_count)
        total_count += 1
        print(count[0])


def load_model():
    model_dir = "checkpoints/upperbound_1/exported_models/saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


def show_inference(model, image_path, writer, count):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    label_names = load_label_names()

    filename = ntpath.basename(image_path)
    image_number = filename.split('.', 1)[0]

    im_width = 256
    im_height = 256

    file_path = os.path.join(save_path, str(image_number) + ".txt")
    file = open(file_path, "w")

    for x in range(0, len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][x] > 0.2:
            ymin = output_dict['detection_boxes'][x][0] * im_width
            xmin = output_dict['detection_boxes'][x][1] * im_height
            ymax = output_dict['detection_boxes'][x][2] * im_width
            xmax = output_dict['detection_boxes'][x][3] * im_height
            label_name = label_names[output_dict['detection_classes'][x] - 1]
            detection_score = output_dict['detection_scores'][x]
            line_of_text = "%s %s %s %s %s %s\n" % (label_name, detection_score, xmin, ymin, xmax, ymax)
            file.write(line_of_text)

        if output_dict['detection_scores'][x] > 0.5:
            writer.addObject(label_name, xmin, ymin, xmax, ymax)
    # writer.save('/Users/chadd/Documents/Chadd/Work/DSO/Model_Re-training/TensorFlow/workspace/training/detected_images/retraining/upperbound_test_xml/' + str(count) + '.xml')

    file.close()

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=1)

    img = Image.fromarray(image_np)
    # img.show()


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == "__main__":
    main()
