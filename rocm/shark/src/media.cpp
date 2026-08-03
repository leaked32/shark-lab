#include "shark/media.hpp"

#include <array>
#include <cstdio>
#include <jpeglib.h>
#include <png.h>
#include <vector>

std::optional<shark::media::texture> shark::media::load_png_texture(
	const std::filesystem::path& path)
{
	FILE* fp = fopen(path.c_str(), "rb");

	if (!fp) {
		return std::nullopt;
	}
	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

	if (!png) {
		fclose(fp);
		return std::nullopt;
	}

	png_infop info = png_create_info_struct(png);

	if (!info) {
		png_destroy_read_struct(&png, nullptr, nullptr);

		fclose(fp);
		return std::nullopt;
	}

	if (setjmp(png_jmpbuf(png))) {
		png_destroy_read_struct(&png, &info, nullptr);

		fclose(fp);
		return std::nullopt;
	}

	png_init_io(png, fp);

	png_read_info(png, info);

	int width = png_get_image_width(png, info);

	int height = png_get_image_height(png, info);

	auto color = png_get_color_type(png, info);

	auto bit_depth = png_get_bit_depth(png, info);

	if (bit_depth == 16) {
		png_set_strip_16(png);
	}
	if (color == PNG_COLOR_TYPE_PALETTE) {
		png_set_palette_to_rgb(png);
	}
	if (color == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
		png_set_expand_gray_1_2_4_to_8(png);
	}
	if (png_get_valid(png, info, PNG_INFO_tRNS)) {
		png_set_tRNS_to_alpha(png);
	}
	if (color == PNG_COLOR_TYPE_RGB || color == PNG_COLOR_TYPE_GRAY) {
		png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
	}

	png_read_update_info(png, info);

	std::vector<unsigned char> pixels(width * height * 4);

	std::vector<png_bytep> rows(height);

	for (int y = 0; y < height; y++) {
		rows[y] = pixels.data() + y * width * 4;
	}

	png_read_image(png, rows.data());

	fclose(fp);

	png_destroy_read_struct(&png, &info, nullptr);

	shark::media::texture result;

	result.width = width;
	result.height = height;

	glGenTextures(1, &result.id);
	glBindTexture(GL_TEXTURE_2D, result.id);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
				 pixels.data());

	return result;
}

std::optional<shark::media::texture> shark::media::load_jpeg_texture(
	const std::filesystem::path& path)
{
	FILE* fp = fopen(path.c_str(), "rb");

	if (!fp) {
		return std::nullopt;
	}

	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_decompress(&cinfo);

	jpeg_stdio_src(&cinfo, fp);

	jpeg_read_header(&cinfo, true);

	jpeg_start_decompress(&cinfo);

	int width = cinfo.output_width;
	int height = cinfo.output_height;
	int channels = cinfo.output_components;

	if (channels != 3) {
		jpeg_destroy_decompress(&cinfo);
		fclose(fp);
		return std::nullopt;
	}

	std::vector<unsigned char> pixels(width * height * 3);

	while (cinfo.output_scanline < height) {
		unsigned char* row = pixels.data() + cinfo.output_scanline * width * 3;

		jpeg_read_scanlines(&cinfo, &row, 1);
	}

	jpeg_finish_decompress(&cinfo);

	jpeg_destroy_decompress(&cinfo);

	fclose(fp);

	shark::media::texture result;

	result.width = width;
	result.height = height;

	glGenTextures(1, &result.id);

	glBindTexture(GL_TEXTURE_2D, result.id);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE,
				 pixels.data());

	return result;
}

std::optional<shark::media::texture> shark::media::load_texture(
	const std::filesystem::path& path)
{
	if (path.extension() == ".png") {
		return shark::media::load_png_texture(path);
	}

	if (path.extension() == ".jpg" || path.extension() == ".jpeg") {
		return shark::media::load_jpeg_texture(path);
	}

	return std::nullopt;
}

#ifdef __linux__

std::string shark::media::find_system_font(
	std::string_view name)
{
	std::string command = "fc-match -f '%{file}' \"" + std::string(name) + "\"";

	std::array<char, 256> buffer = {};

	std::string result;

	FILE* pipe = popen(command.c_str(), "r");

	if (!pipe) {
		return {};
	}

	while (fgets(buffer.data(), buffer.size(), pipe)) {
		result += buffer.data();
	}

	pclose(pipe);

	return result;
}

#endif
