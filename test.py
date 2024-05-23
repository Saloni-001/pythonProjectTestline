import cv2
import pytesseract
from pytesseract import Output
import os


def extract_text_from_image(image_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert the image to RGB (OpenCV uses BGR by default)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use Tesseract to do OCR on the image
        custom_config = r'--oem 3 --psm 6'
        details = pytesseract.image_to_data(rgb_image, output_type=Output.DICT, config=custom_config)

        # Extract text and their bounding box coordinates
        text_elements = []
        for i in range(len(details['text'])):
            if int(details['conf'][i]) > 60:  # Confidence level
                x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
                text = details['text'][i]
                text_elements.append({'text': text, 'bbox': (x, y, w, h)})

        return text_elements
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return []


def segment_visual_elements(image_path, output_dir):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding to segment the image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours of the segments
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        visual_elements = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:  # Filter out small elements
                segment = image[y:y + h, x:x + w]
                segment_path = os.path.join(output_dir, f'element_{i}.png')
                cv2.imwrite(segment_path, segment)
                visual_elements.append(segment_path)

        return visual_elements
    except Exception as e:
        print(f"Error during visual element segmentation: {e}")
        return []


def generate_html(text_elements, visual_elements, output_path='output.html'):
    try:
        html_content = '<html><body>'

        # Add text elements as paragraphs
        for element in text_elements:
            html_content += f'<p>{element["text"]}</p>'

        # Add visual elements as images
        for element in visual_elements:
            html_content += f'<img src="{element}" alt="Visual Element" />'

        html_content += '</body></html>'

        with open(output_path, 'w') as f:
            f.write(html_content)
    except Exception as e:
        print(f"Error generating HTML: {e}")


def main(image_path):
    text_elements = extract_text_from_image(image_path)
    visual_elements = segment_visual_elements(image_path, 'output')
    generate_html(text_elements, visual_elements)


if __name__ == "__main__":
    # Prompt the user to enter the image path
    image_path = input("Enter the path to the image: ").strip()

    # Ensure the provided path is absolute
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)

    main(image_path)
