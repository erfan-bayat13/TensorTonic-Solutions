def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """

    # Step 1: pad the image
    padding = len(kernel) // 2
    padded_image = []

    zero_row = [0] * (len(image[0]) + 2 * padding)

    for _ in range(padding):
        padded_image.append(zero_row.copy())

    for row in image:
        padded_row = [0] * padding + row + [0] * padding
        padded_image.append(padded_row)

    for _ in range(padding):
        padded_image.append(zero_row.copy())

    # Step 2: create output with same dimensions as original
    rows = len(image)
    cols = len(image[0])

    output = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):

            if operation == "erode":
                result = 1

            elif operation == "dilate":
                result = 0

            else:
                raise ValueError("operation must be 'erode' or 'dilate'")

            for ki in range(len(kernel)):
                for kj in range(len(kernel[0])):

                    # Only positions where the kernel is 1 matter
                    if kernel[ki][kj] == 1:
                        pixel = padded_image[row + ki][col + kj]

                        if operation == "erode" and pixel == 0:
                            result = 0
                            break

                        elif operation == "dilate" and pixel == 1:
                            result = 1
                            break

                # Break the outer kernel loop too
                if operation == "erode" and result == 0:
                    break

                if operation == "dilate" and result == 1:
                    break

            output[row][col] = result

    return output