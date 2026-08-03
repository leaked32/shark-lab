#include "shark/media.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <jpeglib.h>
#include <limits>
#include <numeric>
#include <optional>
#include <png.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opusenc.h>
#include <opusfile.h>

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

namespace
{

std::uint16_t u16(
	const std::uint8_t* p)
{
	return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8U);
}

std::uint32_t u32(
	const std::uint8_t* p)
{
	return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8U) |
		(static_cast<std::uint32_t>(p[2]) << 16U) | (static_cast<std::uint32_t>(p[3]) << 24U);
}

std::uint64_t u64(
	const std::uint8_t* p)
{
	return static_cast<std::uint64_t>(u32(p)) | (static_cast<std::uint64_t>(u32(p + 4)) << 32U);
}

void put_u16(
	std::ostream& out, std::uint16_t v)
{
	const std::array<char, 2> b{
		static_cast<char>(v & 0xffU),
		static_cast<char>((v >> 8U) & 0xffU),
	};
	out.write(b.data(), static_cast<std::streamsize>(b.size()));
}

void put_u24(
	std::ostream& out, std::int32_t v)
{
	const auto u = static_cast<std::uint32_t>(v);
	const std::array<char, 3> b{
		static_cast<char>(u & 0xffU),
		static_cast<char>((u >> 8U) & 0xffU),
		static_cast<char>((u >> 16U) & 0xffU),
	};
	out.write(b.data(), static_cast<std::streamsize>(b.size()));
}

void put_u32(
	std::ostream& out, std::uint32_t v)
{
	const std::array<char, 4> b{
		static_cast<char>(v & 0xffU),
		static_cast<char>((v >> 8U) & 0xffU),
		static_cast<char>((v >> 16U) & 0xffU),
		static_cast<char>((v >> 24U) & 0xffU),
	};
	out.write(b.data(), static_cast<std::streamsize>(b.size()));
}

void put_u64(
	std::ostream& out, std::uint64_t v)
{
	put_u32(out, static_cast<std::uint32_t>(v & 0xffffffffULL));
	put_u32(out, static_cast<std::uint32_t>(v >> 32U));
}

bool id_is(
	const std::uint8_t* p, const char (&id)[5])
{
	return std::memcmp(p, id, 4) == 0;
}

std::vector<std::uint8_t> read_all(
	const std::filesystem::path& path)
{
	std::ifstream in(path, std::ios::binary | std::ios::ate);
	if (!in) {
		throw std::runtime_error("cannot open input WAV: " + path.string());
	}
	const auto end = in.tellg();
	if (end < 0) {
		throw std::runtime_error("cannot determine WAV size: " + path.string());
	}
	std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
	in.seekg(0);
	in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	if (!in && !bytes.empty()) {
		throw std::runtime_error("failed to read WAV: " + path.string());
	}
	return bytes;
}

std::int32_t decode_s24(
	const std::uint8_t* p)
{
	std::int32_t value = static_cast<std::int32_t>(p[0]) | (static_cast<std::int32_t>(p[1]) << 8) |
		(static_cast<std::int32_t>(p[2]) << 16);
	if ((value & 0x00800000) != 0) {
		value |= static_cast<std::int32_t>(0xff000000);
	}
	return value;
}

double finite_or_zero(
	double value)
{
	return std::isfinite(value) ? value : 0.0;
}

} // namespace

shark::media::decoded_pcm shark::media::read_wav(
	const std::filesystem::path& path)
{
	const auto bytes = read_all(path);
	if (bytes.size() < 12 || !id_is(bytes.data(), "RIFF") || !id_is(bytes.data() + 8, "WAVE")) {
		throw std::runtime_error("only little-endian RIFF/WAVE files are supported");
	}

	bool have_fmt = false;
	bool have_data = false;
	std::uint16_t format_tag = 0;
	std::uint16_t channels = 0;
	std::uint32_t sample_rate = 0;
	std::uint16_t block_align = 0;
	std::uint16_t bits = 0;
	std::uint16_t valid_bits = 0;
	std::size_t data_offset = 0;
	std::size_t data_size = 0;

	std::size_t offset = 12;
	while (offset + 8 <= bytes.size()) {
		const auto* chunk = bytes.data() + offset;
		const std::uint32_t chunk_size = u32(chunk + 4);
		const std::size_t payload = offset + 8;
		if (payload > bytes.size() || chunk_size > bytes.size() - payload) {
			throw std::runtime_error("truncated WAV chunk");
		}

		if (id_is(chunk, "fmt ")) {
			if (chunk_size < 16) {
				throw std::runtime_error("invalid WAV fmt chunk");
			}
			format_tag = u16(bytes.data() + payload);
			channels = u16(bytes.data() + payload + 2);
			sample_rate = u32(bytes.data() + payload + 4);
			block_align = u16(bytes.data() + payload + 12);
			bits = u16(bytes.data() + payload + 14);
			valid_bits = bits;

			if (format_tag == 0xfffeU) {
				if (chunk_size < 40) {
					throw std::runtime_error("truncated WAVE_FORMAT_EXTENSIBLE fmt chunk");
				}
				valid_bits = u16(bytes.data() + payload + 18);
				format_tag = u16(bytes.data() + payload + 24); // First GUID field: PCM=1, float=3.
			}
			have_fmt = true;
		}
		else if (id_is(chunk, "data")) {
			data_offset = payload;
			data_size = chunk_size;
			have_data = true;
		}

		offset = payload + chunk_size + (chunk_size & 1U);
	}

	if (!have_fmt || !have_data) {
		throw std::runtime_error("WAV is missing fmt or data chunk");
	}
	if (channels == 0 || sample_rate == 0 || block_align == 0 || bits == 0) {
		throw std::runtime_error("invalid WAV format values");
	}
	if (format_tag != 1U && format_tag != 3U) {
		throw std::runtime_error("unsupported WAV encoding; expected PCM integer or IEEE float");
	}

	const std::size_t bytes_per_sample = (bits + 7U) / 8U;
	if (bytes_per_sample == 0 || block_align < channels * bytes_per_sample) {
		throw std::runtime_error("invalid WAV block alignment");
	}
	if (data_size % block_align != 0) {
		throw std::runtime_error("WAV data chunk is not frame-aligned");
	}

	decoded_pcm audio;
	audio.sample_rate = sample_rate;
	audio.channels = channels;
	audio.encoding = format_tag == 1U ? sample_encoding::pcm_integer : sample_encoding::ieee_float;
	audio.bits_per_sample = bits;
	audio.valid_bits_per_sample = valid_bits == 0 ? bits : valid_bits;

	const std::size_t frame_count = data_size / block_align;
	audio.samples.resize(frame_count * channels);

	for (std::size_t frame = 0; frame < frame_count; ++frame) {
		const auto* frame_ptr = bytes.data() + data_offset + frame * block_align;
		for (std::uint16_t channel = 0; channel < channels; ++channel) {
			const auto* p = frame_ptr + channel * bytes_per_sample;
			double value = 0.0;
			if (format_tag == 1U) {
				switch (bits) {
				case 8:
					value = (static_cast<int>(p[0]) - 128) / 128.0;
					break;
				case 16:
					value = static_cast<std::int16_t>(u16(p)) / 32768.0;
					break;
				case 24:
					value = decode_s24(p) / 8388608.0;
					break;
				case 32:
					value = static_cast<std::int32_t>(u32(p)) / 2147483648.0;
					break;
				default:
					throw std::runtime_error("unsupported integer PCM bit depth: " +
											 std::to_string(bits));
				}
			}
			else {
				if (bits == 32) {
					const std::uint32_t raw = u32(p);
					float f = 0.0F;
					std::memcpy(&f, &raw, sizeof(f));
					value = f;
				}
				else if (bits == 64) {
					const std::uint64_t raw = u64(p);
					double d = 0.0;
					std::memcpy(&d, &raw, sizeof(d));
					value = d;
				}
				else {
					throw std::runtime_error("unsupported IEEE-float bit depth: " +
											 std::to_string(bits));
				}
			}
			audio.sample(frame, channel) = finite_or_zero(value);
		}
	}

	return audio;
}

void shark::media::write_wav(
	const std::filesystem::path& path, const decoded_pcm& audio)
{
	if (audio.channels == 0 || audio.sample_rate == 0) {
		throw std::runtime_error("cannot write WAV with zero channels or sample rate");
	}
	const std::size_t frame_count = audio.frames();
	const std::uint16_t bits = audio.bits_per_sample;
	const std::size_t bytes_per_sample = (bits + 7U) / 8U;

	if (audio.encoding == sample_encoding::pcm_integer && bits != 8 && bits != 16 && bits != 24 &&
		bits != 32)
	{
		throw std::runtime_error("unsupported output PCM bit depth");
	}
	if (audio.encoding == sample_encoding::ieee_float && bits != 32 && bits != 64) {
		throw std::runtime_error("unsupported output float bit depth");
	}

	const std::uint64_t block_align_64 =
		static_cast<std::uint64_t>(audio.channels) * bytes_per_sample;
	const std::uint64_t data_size_64 = static_cast<std::uint64_t>(frame_count) * block_align_64;
	if (block_align_64 > std::numeric_limits<std::uint16_t>::max() ||
		data_size_64 > std::numeric_limits<std::uint32_t>::max())
	{
		throw std::runtime_error(
			"output is too large for ordinary RIFF/WAVE; RF64 is not implemented");
	}

	const auto block_align = static_cast<std::uint16_t>(block_align_64);
	const auto data_size = static_cast<std::uint32_t>(data_size_64);
	const std::uint32_t data_padding = data_size & 1U;
	const std::uint32_t riff_size = 36U + data_size + data_padding;

	std::ofstream out(path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("cannot open output WAV: " + path.string());
	}

	out.write("RIFF", 4);
	put_u32(out, riff_size);
	out.write("WAVE", 4);
	out.write("fmt ", 4);
	put_u32(out, 16);
	put_u16(out, audio.encoding == sample_encoding::pcm_integer ? 1U : 3U);
	put_u16(out, audio.channels);
	put_u32(out, audio.sample_rate);
	put_u32(out, audio.sample_rate * block_align);
	put_u16(out, block_align);
	put_u16(out, bits);
	out.write("data", 4);
	put_u32(out, data_size);

	for (double raw_value : audio.samples) {
		const double value = std::clamp(finite_or_zero(raw_value), -1.0, 1.0);
		if (audio.encoding == sample_encoding::pcm_integer) {
			switch (bits) {
			case 8: {
				const auto q = static_cast<std::uint8_t>(
					std::clamp(std::llround(value * 127.5 + 127.5), 0LL, 255LL));
				out.put(static_cast<char>(q));
				break;
			}
			case 16: {
				const auto q = static_cast<std::int16_t>(
					std::clamp(std::llround(value * 32767.0), -32768LL, 32767LL));
				put_u16(out, static_cast<std::uint16_t>(q));
				break;
			}
			case 24: {
				const auto q = static_cast<std::int32_t>(
					std::clamp(std::llround(value * 8388607.0), -8388608LL, 8388607LL));
				put_u24(out, q);
				break;
			}
			case 32: {
				const auto q = static_cast<std::int32_t>(
					std::clamp(std::llround(value * 2147483647.0),
							   static_cast<long long>(std::numeric_limits<std::int32_t>::min()),
							   static_cast<long long>(std::numeric_limits<std::int32_t>::max())));
				put_u32(out, static_cast<std::uint32_t>(q));
				break;
			}
			default:
				throw std::runtime_error("unreachable output PCM bit depth");
			}
		}
		else if (bits == 32) {
			const float f = static_cast<float>(value);
			std::uint32_t raw = 0;
			std::memcpy(&raw, &f, sizeof(raw));
			put_u32(out, raw);
		}
		else {
			const double d = value;
			std::uint64_t raw = 0;
			std::memcpy(&raw, &d, sizeof(raw));
			put_u64(out, raw);
		}
	}

	if (data_padding != 0U) {
		out.put('\0');
	}
	if (!out) {
		throw std::runtime_error("failed while writing WAV: " + path.string());
	}
}

shark::media::decoded_pcm shark::media::read_opus(
	const std::filesystem::path& path)
{
	int error = 0;

	OggOpusFile* file = op_open_file(path.c_str(), &error);

	if (!file) {
		throw std::runtime_error("cannot open opus");
	}

	const int channels = op_channel_count(file, -1);

	const int rate = 48000; // Opus decoder output

	decoded_pcm audio;
	audio.sample_rate = rate;
	audio.channels = channels;
	audio.encoding = sample_encoding::ieee_float;

	std::vector<float> buffer(4096 * channels);

	while (true) {
		int samples = op_read_float(file, buffer.data(), buffer.size(), nullptr);

		if (samples <= 0)
			break;

		for (int i = 0; i < samples * channels; i++) {
			audio.samples.push_back(buffer[i]);
		}
	}

	op_free(file);

	return audio;
}

void shark::media::write_opus(
	const std::filesystem::path& path, const decoded_pcm& audio)
{
	OggOpusComments* comments = ope_comments_create();

	OggOpusEnc* enc = ope_encoder_create_file(path.c_str(), comments, audio.sample_rate,
											  audio.channels, 0, nullptr);

	if (!enc) {
		throw std::runtime_error("cannot create opus");
	}

	std::vector<float> buffer(audio.samples.size());

	for (size_t i = 0; i < audio.samples.size(); i++) {
		buffer[i] = static_cast<float>(audio.samples[i]);
	}

	ope_encoder_write_float(enc, buffer.data(), audio.frames());

	ope_encoder_drain(enc);
	ope_encoder_destroy(enc);

	ope_comments_destroy(comments);
}

shark::media::decoded_pcm shark::media::read_audio(
	const std::filesystem::path& path)
{
	const auto ext = path.extension().string();

	if (ext == ".opus" || ext == ".OPUS") {
		return read_opus(path);
	}
	if (ext == ".wav" || ext == ".WAV") {
		return read_wav(path);
	}

	throw std::runtime_error("unsupported audio format: " + ext);
}

void shark::media::write_audio(
	const std::filesystem::path& path, const decoded_pcm& audio)
{
	const auto ext = path.extension().string();

	if (ext == ".opus" || ext == ".OPUS") {
		shark::media::write_opus(path, audio);
		return;
	}
	if (ext == ".wav" || ext == ".WAV") {
		shark::media::write_wav(path, audio);
		return;
	}

	throw std::runtime_error("unsupported audio format: " + ext);
}

namespace
{

struct Model
{
	std::vector<double> ar; // x[n] + sum(ar[k] * x[n-k]) = error; ar[0] == 1.
	double mean = 0.0;
	double limit = 1.0;
};

double rms_centered(
	const std::vector<double>& x, double mean)
{
	if (x.empty()) {
		return 0.0;
	}
	double sum = 0.0;
	for (double value : x) {
		const double centered = value - mean;
		sum += centered * centered;
	}
	return std::sqrt(sum / static_cast<double>(x.size()));
}

std::optional<Model> fit_model(
	const std::vector<double>& context, std::size_t requested_order, double reflection_limit,
	double prediction_limit_factor)
{
	if (context.size() < 8) {
		return std::nullopt;
	}

	const std::size_t order = std::min(requested_order, context.size() / 4);
	if (order < 2) {
		return std::nullopt;
	}

	const double mean =
		std::accumulate(context.begin(), context.end(), 0.0) / static_cast<double>(context.size());
	std::vector<double> autocorrelation(order + 1, 0.0);
	for (std::size_t lag = 0; lag <= order; ++lag) {
		double sum = 0.0;
		for (std::size_t i = lag; i < context.size(); ++i) {
			sum += (context[i] - mean) * (context[i - lag] - mean);
		}
		autocorrelation[lag] = sum / static_cast<double>(context.size());
	}

	if (!(autocorrelation[0] > 1e-14) || !std::isfinite(autocorrelation[0])) {
		Model constant;
		constant.ar = {1.0};
		constant.mean = mean;
		constant.limit = std::max(1e-6, std::abs(mean) * 2.0);
		return constant;
	}

	autocorrelation[0] *= 1.000001; // Tiny regularization for nearly periodic contexts.
	std::vector<double> ar(order + 1, 0.0);
	ar[0] = 1.0;
	double error = autocorrelation[0];

	for (std::size_t i = 1; i <= order; ++i) {
		double numerator = autocorrelation[i];
		for (std::size_t j = 1; j < i; ++j) {
			numerator += ar[j] * autocorrelation[i - j];
		}

		double reflection = -numerator / std::max(error, 1e-18);
		reflection = std::clamp(reflection, -reflection_limit, reflection_limit);

		const auto previous = ar;
		for (std::size_t j = 1; j < i; ++j) {
			ar[j] = previous[j] + reflection * previous[i - j];
		}
		ar[i] = reflection;
		error *= std::max(1e-6, 1.0 - reflection * reflection);
		if (!std::isfinite(error)) {
			return std::nullopt;
		}
	}

	const double context_rms = rms_centered(context, mean);
	double peak = 0.0;
	for (double value : context) {
		peak = std::max(peak, std::abs(value - mean));
	}

	Model model;
	model.ar = std::move(ar);
	model.mean = mean;
	model.limit = std::max({1e-5, context_rms * prediction_limit_factor, peak * 1.25});
	return model;
}

std::vector<double> predict(
	const std::vector<double>& context, std::size_t count,
	const shark::media::declick::lpc_options& config)
{
	std::vector<double> result(count, 0.0);
	const auto model =
		fit_model(context, config.order, config.reflection_limit, config.prediction_limit_factor);
	if (!model) {
		return {};
	}
	if (model->ar.size() == 1) {
		std::fill(result.begin(), result.end(), model->mean);
		return result;
	}

	const std::size_t order = model->ar.size() - 1;
	std::vector<double> history;
	history.reserve(order + count);
	const std::size_t first = context.size() > order ? context.size() - order : 0;
	for (std::size_t i = first; i < context.size(); ++i) {
		history.push_back(context[i] - model->mean);
	}
	while (history.size() < order) {
		history.insert(history.begin(), 0.0);
	}

	for (std::size_t n = 0; n < count; ++n) {
		double value = 0.0;
		for (std::size_t k = 1; k <= order; ++k) {
			value -= model->ar[k] * history[history.size() - k];
		}
		if (!std::isfinite(value)) {
			value = 0.0;
		}
		value = std::clamp(value, -model->limit, model->limit);
		history.push_back(value);
		result[n] = value + model->mean;
	}
	return result;
}

std::vector<double> cubic_bridge(
	const std::vector<double>& signal, std::size_t begin, std::size_t end)
{
	const std::size_t count = end - begin;
	std::vector<double> result(count, 0.0);
	if (count == 0) {
		return result;
	}

	const bool have_left = begin > 0;
	const bool have_right = end < signal.size();
	const double left = have_left ? signal[begin - 1] : (have_right ? signal[end] : 0.0);
	const double right = have_right ? signal[end] : left;
	const double scale = static_cast<double>(count + 1);

	// Zero-slope cubic smoothstep. It cannot overshoot the two boundary values,
	// which is crucial when repairing already clipped microphone overloads.
	for (std::size_t i = 0; i < count; ++i) {
		const double t = static_cast<double>(i + 1) / scale;
		const double w = t * t * (3.0 - 2.0 * t);
		result[i] = left * (1.0 - w) + right * w;
	}
	return result;
}

double smoothstep(
	double x)
{
	x = std::clamp(x, 0.0, 1.0);
	return x * x * (3.0 - 2.0 * x);
}

struct SignalStats
{
	double peak = 0.0;
	double rms = 0.0;
	double derivative_rms = 0.0;
};

SignalStats statistics(
	const std::vector<double>& x)
{
	SignalStats stats;
	if (x.empty()) {
		return stats;
	}
	double energy = 0.0;
	double derivative_energy = 0.0;
	for (std::size_t i = 0; i < x.size(); ++i) {
		stats.peak = std::max(stats.peak, std::abs(x[i]));
		energy += x[i] * x[i];
		if (i > 0) {
			const double d = x[i] - x[i - 1];
			derivative_energy += d * d;
		}
	}
	stats.rms = std::sqrt(energy / static_cast<double>(x.size()));
	stats.derivative_rms =
		x.size() > 1 ? std::sqrt(derivative_energy / static_cast<double>(x.size() - 1)) : 0.0;
	return stats;
}

std::vector<double> nearby_context(
	const std::vector<double>& signal, std::size_t begin, std::size_t end,
	std::size_t context_frames)
{
	const std::size_t left_begin = begin > context_frames ? begin - context_frames : 0;
	const std::size_t right_end = std::min(signal.size(), end + context_frames);
	std::vector<double> context;
	context.reserve((begin - left_begin) + (right_end - end));
	context.insert(context.end(), signal.begin() + static_cast<std::ptrdiff_t>(left_begin),
				   signal.begin() + static_cast<std::ptrdiff_t>(begin));
	context.insert(context.end(), signal.begin() + static_cast<std::ptrdiff_t>(end),
				   signal.begin() + static_cast<std::ptrdiff_t>(right_end));
	return context;
}

} // namespace

std::vector<double> shark::media::declick::reconstruct_gap_cubic(
	const std::vector<double>& signal, std::size_t begin, std::size_t end)
{
	if (begin >= end || end > signal.size()) {
		return {};
	}
	return cubic_bridge(signal, begin, end);
}

std::vector<double> shark::media::declick::reconstruct_gap_lpc(
	const std::vector<double>& signal, std::size_t begin, std::size_t end,
	const lpc_options& config)
{
	if (begin >= end || end > signal.size()) {
		return {};
	}

	const std::size_t count = end - begin;
	const std::size_t left_begin =
		begin > config.context_frames ? begin - config.context_frames : 0;
	const std::size_t right_end = std::min(signal.size(), end + config.context_frames);

	std::vector<double> left(signal.begin() + static_cast<std::ptrdiff_t>(left_begin),
							 signal.begin() + static_cast<std::ptrdiff_t>(begin));
	std::vector<double> right(signal.begin() + static_cast<std::ptrdiff_t>(end),
							  signal.begin() + static_cast<std::ptrdiff_t>(right_end));

	auto forward = predict(left, count, config);
	std::reverse(right.begin(), right.end());
	auto backward_reversed = predict(right, count, config);
	std::reverse(backward_reversed.begin(), backward_reversed.end());

	if (forward.empty() && backward_reversed.empty()) {
		return cubic_bridge(signal, begin, end);
	}
	if (forward.empty()) {
		return backward_reversed;
	}
	if (backward_reversed.empty()) {
		return forward;
	}

	std::vector<double> result(count, 0.0);
	for (std::size_t i = 0; i < count; ++i) {
		const double t = smoothstep(static_cast<double>(i + 1) / static_cast<double>(count + 1));
		result[i] = forward[i] * (1.0 - t) + backward_reversed[i] * t;
	}
	return result;
}

std::vector<double> shark::media::declick::reconstruct_gap_hybrid(
	const std::vector<double>& signal, std::size_t begin, std::size_t end,
	const lpc_options& config)
{
	auto cubic = reconstruct_gap_cubic(signal, begin, end);
	auto lpc = reconstruct_gap_lpc(signal, begin, end, config);
	if (cubic.empty() || lpc.size() != cubic.size()) {
		return cubic;
	}

	const auto context = nearby_context(signal, begin, end, config.context_frames);
	const SignalStats context_stats = statistics(context);
	const SignalStats lpc_stats = statistics(lpc);

	const double peak_limit = std::max(0.08, context_stats.peak * 1.75);
	const double rms_limit = std::max(0.025, context_stats.rms * 2.5);
	const double derivative_limit = std::max(0.015, context_stats.derivative_rms * 2.5);
	if (lpc_stats.peak > peak_limit || lpc_stats.rms > rms_limit ||
		lpc_stats.derivative_rms > derivative_limit)
	{
		return cubic;
	}

	const double mix = std::clamp(config.hybrid_mix, 0.0, 1.0);
	for (std::size_t i = 0; i < cubic.size(); ++i) {
		cubic[i] = cubic[i] * (1.0 - mix) + lpc[i] * mix;
	}
	return cubic;
}

namespace
{

std::size_t ms_to_frames(
	double milliseconds, std::uint32_t sample_rate)
{
	return static_cast<std::size_t>(std::llround(milliseconds * sample_rate / 1000.0));
}

double median_abs(
	std::vector<double> values)
{
	if (values.empty()) {
		return 0.0;
	}
	for (double& value : values) {
		value = std::abs(value);
	}
	const std::size_t middle = values.size() / 2;
	std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
					 values.end());
	double result = values[middle];
	if (values.size() % 2 == 0 && middle > 0) {
		const auto lower =
			std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
		result = (*lower + result) * 0.5;
	}
	return result;
}

// double smoothstep(
// 	double x)
// {
// 	x = std::clamp(x, 0.0, 1.0);
// 	return x * x * (3.0 - 2.0 * x);
// }

shark::media::declick::event_kind merge_kind(
	shark::media::declick::event_kind a, shark::media::declick::event_kind b)
{
	return a == b ? a : shark::media::declick::event_kind::mixed;
}

struct Candidate
{
	std::size_t core_begin = 0;
	std::size_t core_end = 0;
	std::size_t repair_begin = 0;
	std::size_t repair_end = 0;
	shark::media::declick::event_kind kind = shark::media::declick::event_kind::impulse;
	double score = 0.0;
};

std::vector<Candidate> detect_channel(
	const std::vector<double>& signal, std::uint32_t sample_rate,
	const shark::media::declick::options& config, std::size_t& clipped_seed_frames,
	std::size_t& impulse_seed_frames)
{
	const std::size_t n = signal.size();
	if (n < 3) {
		return {};
	}

	std::vector<double> derivative(n, 0.0);
	for (std::size_t i = 1; i < n; ++i) {
		derivative[i] = signal[i] - signal[i - 1];
	}

	const double global_scale = std::max(1e-7, median_abs(derivative) * 1.4826);
	std::vector<double> squared_prefix(n + 1, 0.0);
	for (std::size_t i = 0; i < n; ++i) {
		squared_prefix[i + 1] = squared_prefix[i] + derivative[i] * derivative[i];
	}

	const std::size_t baseline_window =
		std::max<std::size_t>(1, ms_to_frames(config.baseline_window_ms, sample_rate));
	const std::size_t exclusion = ms_to_frames(config.baseline_exclusion_ms, sample_rate);
	std::vector<double> scores(n, 0.0);
	std::vector<unsigned char> clipped(n, 0);
	std::vector<unsigned char> impulse(n, 0);
	std::vector<unsigned char> seed(n, 0);

	for (std::size_t i = 0; i < n; ++i) {
		const std::size_t left_begin = i > baseline_window ? i - baseline_window : 0;
		const std::size_t left_end = i > exclusion ? i - exclusion : 0;
		const std::size_t right_begin = std::min(n, i + exclusion);
		const std::size_t right_end = std::min(n, i + baseline_window);

		double sum = 0.0;
		std::size_t count = 0;
		if (left_end > left_begin) {
			sum += squared_prefix[left_end] - squared_prefix[left_begin];
			count += left_end - left_begin;
		}
		if (right_end > right_begin) {
			sum += squared_prefix[right_end] - squared_prefix[right_begin];
			count += right_end - right_begin;
		}

		const double local_rms =
			count > 0 ? std::sqrt(sum / static_cast<double>(count)) : global_scale;
		const double baseline = std::max(global_scale, local_rms);
		scores[i] = std::abs(derivative[i]) / std::max(1e-12, baseline);

		clipped[i] = static_cast<unsigned char>(config.detect_clipping &&
												std::abs(signal[i]) >= config.clip_level);
		impulse[i] =
			static_cast<unsigned char>(config.detect_impulses &&
									   ((std::abs(derivative[i]) >= config.derivative_floor &&
										 scores[i] >= config.derivative_ratio) ||
										std::abs(derivative[i]) >= config.hard_derivative));
		seed[i] = static_cast<unsigned char>(clipped[i] || impulse[i]);
		clipped_seed_frames += clipped[i] != 0;
		impulse_seed_frames += impulse[i] != 0;
	}

	const std::size_t max_gap = ms_to_frames(config.merge_gap_ms, sample_rate);
	std::vector<Candidate> candidates;
	std::size_t i = 0;
	while (i < n) {
		while (i < n && seed[i] == 0) {
			++i;
		}
		if (i == n) {
			break;
		}

		const std::size_t begin = i;
		std::size_t end = i;
		std::size_t last_seed = i;
		bool has_clip = clipped[i] != 0;
		bool has_impulse = impulse[i] != 0;
		double max_score = scores[i];
		++i;

		while (i < n) {
			if (seed[i] != 0) {
				if (i - last_seed > max_gap + 1) {
					break;
				}
				last_seed = i;
				end = i;
				has_clip = has_clip || clipped[i] != 0;
				has_impulse = has_impulse || impulse[i] != 0;
				max_score = std::max(max_score, scores[i]);
			}
			else if (i - last_seed > max_gap) {
				break;
			}
			++i;
		}

		Candidate candidate;
		candidate.core_begin = begin;
		candidate.core_end = end;
		candidate.kind = has_clip && has_impulse ? shark::media::declick::event_kind::mixed
			: has_clip							 ? shark::media::declick::event_kind::clipped
												 : shark::media::declick::event_kind::impulse;
		candidate.score = max_score;
		candidates.push_back(candidate);
	}

	const std::size_t pre_roll = ms_to_frames(config.pre_roll_ms, sample_rate);
	const std::size_t post_roll = ms_to_frames(config.post_roll_ms, sample_rate);
	const std::size_t max_repair =
		std::max<std::size_t>(1, ms_to_frames(config.max_repair_ms, sample_rate));

	for (auto& candidate : candidates) {
		candidate.repair_begin =
			candidate.core_begin > pre_roll ? candidate.core_begin - pre_roll : 0;
		candidate.repair_end = std::min(n - 1, candidate.core_end + post_roll);

		const std::size_t repair_length = candidate.repair_end - candidate.repair_begin + 1;
		const std::size_t core_length = candidate.core_end - candidate.core_begin + 1;
		if (repair_length > max_repair && core_length < max_repair) {
			const std::size_t spare = max_repair - core_length;
			const std::size_t left = std::min(candidate.core_begin, spare / 2);
			const std::size_t right = std::min(n - 1 - candidate.core_end, spare - left);
			candidate.repair_begin = candidate.core_begin - left;
			candidate.repair_end = candidate.core_end + right;
		}
	}

	std::vector<Candidate> merged;
	for (const auto& candidate : candidates) {
		if (merged.empty() || candidate.repair_begin > merged.back().repair_end + 1) {
			merged.push_back(candidate);
			continue;
		}
		auto& previous = merged.back();
		previous.core_begin = std::min(previous.core_begin, candidate.core_begin);
		previous.core_end = std::max(previous.core_end, candidate.core_end);
		previous.repair_begin = std::min(previous.repair_begin, candidate.repair_begin);
		previous.repair_end = std::max(previous.repair_end, candidate.repair_end);
		previous.kind = merge_kind(previous.kind, candidate.kind);
		previous.score = std::max(previous.score, candidate.score);
	}
	return merged;
}

double peak_in_range(
	const std::vector<double>& signal, std::size_t begin, std::size_t end)
{
	double peak = 0.0;
	for (std::size_t i = begin; i <= end; ++i) {
		peak = std::max(peak, std::abs(signal[i]));
	}
	return peak;
}

} // namespace

shark::media::declick::options shark::media::declick::preset_config(
	const std::string& name)
{
	options config;
	if (name == "asmr") {
		return config;
	}
	if (name == "aggressive") {
		config.repair_mode = repair_mode::cubic;
		config.clip_level = 0.98;
		config.derivative_floor = 0.025;
		config.hard_derivative = 0.10;
		config.derivative_ratio = 8.0;
		config.merge_gap_ms = 2.0;
		config.pre_roll_ms = 2.0;
		config.post_roll_ms = 4.0;
		config.max_repair_ms = 80.0;
		config.lpc_order = 40;
		config.lpc_context_ms = 30.0;
		return config;
	}
	if (name == "conservative") {
		config.repair_mode = repair_mode::hybrid;
		config.lpc_mix = 0.35;
		config.derivative_floor = 0.10;
		config.hard_derivative = 0.25;
		config.derivative_ratio = 13.0;
		config.merge_gap_ms = 1.0;
		config.pre_roll_ms = 1.0;
		config.post_roll_ms = 2.0;
		config.max_repair_ms = 40.0;
		config.lpc_order = 24;
		return config;
	}
	if (name == "clip-only") {
		config.repair_mode = repair_mode::cubic;
		config.detect_impulses = false;
		config.clip_level = 0.985;
		config.pre_roll_ms = 1.5;
		config.post_roll_ms = 3.0;
		return config;
	}
	throw std::invalid_argument("unknown preset: " + name);
}

shark::media::declick::Result shark::media::declick::process(
	shark::media::decoded_pcm& audio, const options& config)
{
	if (audio.channels == 0 || audio.sample_rate == 0) {
		throw std::invalid_argument("invalid audio buffer");
	}

	Result result;
	const std::size_t frame_count = audio.frames();
	for (std::uint16_t channel = 0; channel < audio.channels; ++channel) {
		std::vector<double> signal(frame_count, 0.0);
		for (std::size_t frame = 0; frame < frame_count; ++frame) {
			signal[frame] = audio.sample(frame, channel);
		}

		auto candidates = detect_channel(signal, audio.sample_rate, config,
										 result.clipped_seed_frames, result.impulse_seed_frames);

		lpc_options lpc;
		lpc.order = config.lpc_order;
		lpc.context_frames = std::max<std::size_t>(
			config.lpc_order * 4, ms_to_frames(config.lpc_context_ms, audio.sample_rate));
		lpc.reflection_limit = config.reflection_limit;
		lpc.prediction_limit_factor = config.prediction_limit_factor;
		lpc.hybrid_mix = config.lpc_mix;

		for (const auto& candidate : candidates) {
			const double original_peak =
				peak_in_range(signal, candidate.repair_begin, candidate.repair_end);
			std::vector<double> replacement;
			const bool force_cubic = candidate.kind != event_kind::impulse;
			if (force_cubic || config.repair_mode == repair_mode::cubic) {
				replacement =
					reconstruct_gap_cubic(signal, candidate.repair_begin, candidate.repair_end + 1);
			}
			else if (config.repair_mode == repair_mode::hybrid) {
				replacement = reconstruct_gap_hybrid(signal, candidate.repair_begin,
													 candidate.repair_end + 1, lpc);
			}
			else {
				replacement = reconstruct_gap_lpc(signal, candidate.repair_begin,
												  candidate.repair_end + 1, lpc);
			}
			if (replacement.size() != candidate.repair_end - candidate.repair_begin + 1) {
				continue;
			}

			for (std::size_t frame = candidate.repair_begin; frame <= candidate.repair_end; ++frame)
			{
				double weight = 1.0;
				if (frame < candidate.core_begin) {
					const double numerator =
						static_cast<double>(frame - candidate.repair_begin + 1);
					const double denominator =
						static_cast<double>(candidate.core_begin - candidate.repair_begin + 1);
					weight = std::min(weight, smoothstep(numerator / denominator));
				}
				if (frame > candidate.core_end) {
					const double numerator = static_cast<double>(candidate.repair_end - frame + 1);
					const double denominator =
						static_cast<double>(candidate.repair_end - candidate.core_end + 1);
					weight = std::min(weight, smoothstep(numerator / denominator));
				}
				const std::size_t index = frame - candidate.repair_begin;
				signal[frame] = signal[frame] * (1.0 - weight) + replacement[index] * weight;
			}

			event e;
			e.channel = channel;
			e.core_begin = candidate.core_begin;
			e.core_end = candidate.core_end;
			e.repair_begin = candidate.repair_begin;
			e.repair_end = candidate.repair_end;
			e.kind = candidate.kind;
			e.score = candidate.score;
			e.original_peak = original_peak;
			e.repaired_peak = peak_in_range(signal, candidate.repair_begin, candidate.repair_end);
			result.events.push_back(e);
		}

		for (std::size_t frame = 0; frame < frame_count; ++frame) {
			audio.sample(frame, channel) = signal[frame];
		}
	}

	std::sort(result.events.begin(), result.events.end(),
			  [](const event& a, const event& b)
			  {
				  if (a.repair_begin != b.repair_begin) {
					  return a.repair_begin < b.repair_begin;
				  }
				  return a.channel < b.channel;
			  });
	return result;
}

const char* shark::media::declick::to_string(
	event_kind kind)
{
	switch (kind) {
	case event_kind::clipped:
		return "clipped";
	case event_kind::impulse:
		return "impulse";
	case event_kind::mixed:
		return "mixed";
	}
	return "unknown";
}

const char* shark::media::declick::to_string(
	repair_mode mode)
{
	switch (mode) {
	case repair_mode::cubic:
		return "cubic";
	case repair_mode::hybrid:
		return "hybrid";
	case repair_mode::lpc:
		return "lpc";
	}
	return "unknown";
}

shark::media::declick::repair_mode shark::media::declick::parse_repair_mode(
	const std::string& name)
{
	if (name == "cubic") {
		return repair_mode::cubic;
	}
	if (name == "hybrid") {
		return repair_mode::hybrid;
	}
	if (name == "lpc") {
		return repair_mode::lpc;
	}
	throw std::invalid_argument("unknown repair mode: " + name);
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
