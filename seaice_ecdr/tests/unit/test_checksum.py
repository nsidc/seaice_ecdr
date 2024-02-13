from seaice_ecdr.checksum import _get_checksum, write_checksum_file


def test_write_checksum_file(tmp_path):
    example_file = tmp_path / "foo.nc"

    some_completely_random_data = b"some completely random data"
    with open(example_file, "wb") as mock_nc_file:
        mock_nc_file.write(some_completely_random_data)

    size_in_bytes = example_file.stat().st_size
    checksum_of_data = _get_checksum(example_file)

    checksum_file = write_checksum_file(
        input_filepath=example_file,
        ecdr_data_dir=tmp_path,
        product_type="complete_daily",
    )

    assert checksum_file.is_file()
    # The checksum file should be the input file's name + ".mnf".
    assert checksum_file.name == example_file.name + ".mnf"

    assert (
        checksum_file.read_text()
        == f"{example_file.name},{checksum_of_data},{size_in_bytes}"
    )
