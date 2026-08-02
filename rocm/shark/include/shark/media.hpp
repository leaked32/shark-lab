#pragma once

#include <GL/gl.h>
#include <filesystem>
#include <optional>

namespace shark::media
{

struct texture
{
	GLuint id = 0;
	int width = 0;
	int height = 0;
};

std::optional<texture> load_texture(const std::filesystem::path& path);
std::optional<texture> load_png_texture(const std::filesystem::path& path);
std::optional<texture> load_jpeg_texture(const std::filesystem::path& path);

std::string find_system_font(std::string_view name);
} // namespace shark::media
