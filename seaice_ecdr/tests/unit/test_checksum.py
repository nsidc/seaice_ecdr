from seaice_ecdr.checksum import _get_checksum, write_checksum_file


def test_write_checksum_file(tmp_path):
    # Write mock data to a temporary file.
    example_file = tmp_path / "foo.nc"
    with open(example_file, "wb") as mock_nc_file:
        mock_nc_file.write(b"some completely random data")

    # Get the size and checksum.
    size_in_bytes = example_file.stat().st_size
    checksum_of_data = _get_checksum(example_file)

    # Use the function being tested to create a checksum file.
    checksum_file = write_checksum_file(
        input_filepath=example_file,
        output_dir=tmp_path,
    )

    # Assert that it was written.
    assert checksum_file.is_file()
    # The checksum file should be the input file's name + ".mnf".
    assert checksum_file.name == example_file.name + ".mnf"

    # The contents should be the a single line of comma-separated-values:
    # {input_filename},{checksum},{size_in_bytes}`
    assert (
        checksum_file.read_text()
        == f"{example_file.name},{checksum_of_data},{size_in_bytes}"
    )
