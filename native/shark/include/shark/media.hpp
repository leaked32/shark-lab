/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: shark/include/shark/media.hpp
 *
 * License: MIT
 */

#pragma once

#ifndef __cplusplus
#error "This header requires C++"
#endif

#include <GL/gl.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

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

enum class sample_encoding {
	pcm_integer,
	ieee_float,
};

// decoded_pcm is not a WAV abstraction. It is a decoded PCM audio representation
struct decoded_pcm
{
	std::uint32_t sample_rate = 0;
	std::uint16_t channels = 0;
	sample_encoding encoding = sample_encoding::pcm_integer;
	std::uint16_t bits_per_sample = 24;
	std::uint16_t valid_bits_per_sample = 24;
	std::vector<double> samples; // Interleaved, normalized to approximately [-1, 1].

	[[nodiscard]] std::size_t frames() const
	{
		if (channels == 0) {
			return 0;
		}
		if (samples.size() % channels != 0) {
			throw std::runtime_error("interleaved sample count is not divisible by channel count");
		}
		return samples.size() / channels;
	}

	[[nodiscard]] double sample(
		std::size_t frame, std::uint16_t channel) const
	{
		return samples.at(frame * channels + channel);
	}

	double& sample(
		std::size_t frame, std::uint16_t channel)
	{
		return samples.at(frame * channels + channel);
	}
};

decoded_pcm read_wav(const std::filesystem::path& path);
void write_wav(const std::filesystem::path& path, const decoded_pcm& audio);

// sudo apt install libopusfile-dev
// sudo apt install libopusenc-dev
decoded_pcm read_opus(const std::filesystem::path& path);
void write_opus(const std::filesystem::path& path, const decoded_pcm& audio);

// Sequential Opus I/O.  These keep only the caller's current block in memory.
class opus_reader
{
  public:
	explicit opus_reader(const std::filesystem::path& path);
	~opus_reader();
	opus_reader(const opus_reader&) = delete;
	opus_reader& operator=(const opus_reader&) = delete;

	[[nodiscard]] const decoded_pcm& format() const;
	[[nodiscard]] std::size_t total_frames() const;
	[[nodiscard]] decoded_pcm read_frames(std::size_t maximum_frames);

  private:
	struct state;
	std::unique_ptr<state> state_;
};

class opus_writer
{
  public:
	opus_writer(const std::filesystem::path& path, const decoded_pcm& format);
	~opus_writer();
	opus_writer(const opus_writer&) = delete;
	opus_writer& operator=(const opus_writer&) = delete;

	void write_frames(const decoded_pcm& audio);
	void finish();

  private:
	struct state;
	std::unique_ptr<state> state_;
};

decoded_pcm read_audio(const std::filesystem::path& path);
void write_audio(const std::filesystem::path& path, const decoded_pcm& audio);

// declicker specialized for ambient sounds.
namespace declick
{

struct lpc_options
{
	std::size_t order = 32;
	std::size_t context_frames = 1103; // About 25 ms at 44.1 kHz.
	double reflection_limit = 0.985;
	double prediction_limit_factor = 4.0;
	double hybrid_mix = 0.25;
};

// Returns replacements for signal[begin, end), aligned left-to-right.
std::vector<double> reconstruct_gap_cubic(const std::vector<double>& signal, std::size_t begin,
										  std::size_t end);

std::vector<double> reconstruct_gap_lpc(const std::vector<double>& signal, std::size_t begin,
										std::size_t end, const lpc_options& config);

std::vector<double> reconstruct_gap_hybrid(const std::vector<double>& signal, std::size_t begin,
										   std::size_t end, const lpc_options& config);

enum class repair_mode {
	cubic,
	hybrid,
	lpc,
};

enum class event_kind {
	clipped,
	impulse,
	mixed,
};

struct event
{
	std::uint16_t channel = 0;
	std::size_t core_begin = 0;
	std::size_t core_end = 0; // Inclusive.
	std::size_t repair_begin = 0;
	std::size_t repair_end = 0; // Inclusive.
	event_kind kind = event_kind::impulse;
	double score = 0.0;
	double original_peak = 0.0;
	double repaired_peak = 0.0;
};

struct options
{
	bool detect_clipping = true;
	bool detect_impulses = true;

	double clip_level = 0.985;
	double derivative_floor = 0.05;
	double hard_derivative = 0.18;
	double derivative_ratio = 10.0;
	double baseline_window_ms = 20.0;
	double baseline_exclusion_ms = 2.0;

	double merge_gap_ms = 1.5;
	double pre_roll_ms = 1.5;
	double post_roll_ms = 3.0;
	double max_repair_ms = 60.0;

	repair_mode repair_mode = repair_mode::hybrid;
	std::size_t lpc_order = 32;
	double lpc_context_ms = 25.0;
	double reflection_limit = 0.985;
	double prediction_limit_factor = 4.0;
	double lpc_mix = 0.25;
	std::size_t worker_threads = 0;
};

struct Result
{
	std::vector<event> events;
	std::size_t clipped_seed_frames = 0;
	std::size_t impulse_seed_frames = 0;
};

options preset_config(const std::string& name);
Result process(shark::media::decoded_pcm& audio, const options& config);
const char* to_string(event_kind kind);
const char* to_string(repair_mode mode);
repair_mode parse_repair_mode(const std::string& name);

} // namespace declick

std::string find_system_font(std::string_view name);
} // namespace shark::media
