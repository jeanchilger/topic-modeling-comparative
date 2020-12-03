import csv

def matrix_to_csv(matrix, output_file_path, header=None):
    """
    Writes the given matrix into an csv file.
    Each row in matrix corresponds to a row in the
    file. Optionally a list may be given as a header.

    Args:
        matrix (list of list): Input matrix to be written.
        output_file_path (str): Path to the destination file.
        header (list, optional): Header for the resulting csv. 
          Defaults to None.
    """

    with open(output_file_path, "w") as dest_csv:
        writer = csv.writer(dest_csv)

        if header:
            writer.writerow(header)

        for row in matrix:
            writer.writerow(row)

