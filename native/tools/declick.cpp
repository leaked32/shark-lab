#include "shark/media.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace lo = shark::media;
namespace sd = lo::declick;

namespace
{

struct options
{
	std::string preset = "aggressive";
	std::filesystem::path input;
	std::filesystem::path output;
	std::filesystem::path report;
	bool dry_run = false;
	double chunk_seconds = 120.0;
	std::size_t threads = 0;
	std::vector<std::pair<std::string, std::string>> overrides;
};

void usage(
	std::ostream& out)
{
	constexpr const char* use =
		"shark-declick: aggressive short-pulse and clipping repair for WAV audio\n\n"
		"usage:\n"
		"  shark-declick [options] input.wav output.wav\n\n"
		"examples:\n"
		"  shark-declick --preset aggressive input.wav output.wav\n\n"
		"  shark-declick --preset aggressive input.opus output.opus\n\n"
		"options:\n"
		"  --preset NAME              aggressive (default), asmr, conservative, clip-only\n"
		"  --clip-level X             full-scale seed threshold, default 0.985\n"
		"  --derivative-floor X       minimum jump used with local-ratio test\n"
		"  --hard-derivative X        always detect jumps at least this large\n"
		"  --derivative-ratio X       jump / robust local baseline threshold\n"
		"  --merge-gap-ms X           merge nearby seed samples into one pulse\n"
		"  --pre-roll-ms X            context replaced before detected pulse\n"
		"  --post-roll-ms X           context replaced after detected pulse\n"
		"  --max-repair-ms X          cap ordinary repair-region length\n"
		"  --repair-mode NAME         cubic, hybrid, or lpc\n"
		"  --lpc-order N              linear-prediction order\n"
		"  --lpc-context-ms X         context used on each side\n"
		"  --lpc-mix X                LPC share in hybrid mode, 0..1\n"
		"  --report FILE.csv          write detected-event diagnostics\n"
		"  --chunk-seconds X          streamed core block duration, default 120\n"
		"  --threads N                channel worker threads; 0 = auto (default)\n"
		"  --dry-run                  detect/report without writing output\n"
		"  -h, --help                 show this message\n";
	out << use;
}

double parse_double(
	const std::string& option, const std::string& value)
{
	std::size_t consumed = 0;
	const double result = std::stod(value, &consumed);
	if (consumed != value.size()) {
		throw std::invalid_argument("invalid value for " + option + ": " + value);
	}
	return result;
}

std::size_t parse_size(
	const std::string& option, const std::string& value)
{
	std::size_t consumed = 0;
	const auto result = std::stoull(value, &consumed);
	if (consumed != value.size()) {
		throw std::invalid_argument("invalid value for " + option + ": " + value);
	}
	return static_cast<std::size_t>(result);
}

options parse_options(
	int argc, char** argv)
{
	options options;
	std::vector<std::string> positional;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "-h" || arg == "--help") {
			usage(std::cout);
			std::exit(0);
		}
		if (arg == "--dry-run") {
			options.dry_run = true;
			continue;
		}
		if (arg.rfind("--", 0) == 0) {
			if (i + 1 >= argc) {
				throw std::invalid_argument("missing value after " + arg);
			}
			const std::string value = argv[++i];
			if (arg == "--preset") {
				options.preset = value;
			}
			else if (arg == "--report") {
				options.report = value;
			}
			else if (arg == "--chunk-seconds") {
				options.chunk_seconds = parse_double(arg, value);
			}
			else if (arg == "--threads") {
				options.threads = parse_size(arg, value);
			}
			else {
				options.overrides.emplace_back(arg, value);
			}
			continue;
		}
		positional.push_back(arg);
	}

	if (options.dry_run) {
		if (positional.size() != 1 && positional.size() != 2) {
			throw std::invalid_argument("dry-run expects input.wav and optionally output.wav");
		}
	}
	else if (positional.size() != 2) {
		throw std::invalid_argument("expected input.wav and output.wav");
	}

	options.input = positional[0];
	if (positional.size() == 2) {
		options.output = positional[1];
	}
	return options;
}

void apply_overrides(
	sd::options& config, const options& options)
{
	for (const auto& [name, value] : options.overrides) {
		if (name == "--clip-level") {
			config.clip_level = parse_double(name, value);
		}
		else if (name == "--derivative-floor") {
			config.derivative_floor = parse_double(name, value);
		}
		else if (name == "--hard-derivative") {
			config.hard_derivative = parse_double(name, value);
		}
		else if (name == "--derivative-ratio") {
			config.derivative_ratio = parse_double(name, value);
		}
		else if (name == "--merge-gap-ms") {
			config.merge_gap_ms = parse_double(name, value);
		}
		else if (name == "--pre-roll-ms") {
			config.pre_roll_ms = parse_double(name, value);
		}
		else if (name == "--post-roll-ms") {
			config.post_roll_ms = parse_double(name, value);
		}
		else if (name == "--max-repair-ms") {
			config.max_repair_ms = parse_double(name, value);
		}
		else if (name == "--repair-mode") {
			config.repair_mode = sd::parse_repair_mode(value);
		}
		else if (name == "--lpc-order") {
			config.lpc_order = parse_size(name, value);
		}
		else if (name == "--lpc-context-ms") {
			config.lpc_context_ms = parse_double(name, value);
		}
		else if (name == "--lpc-mix") {
			config.lpc_mix = parse_double(name, value);
		}
		else {
			throw std::invalid_argument("unknown option: " + name);
		}
	}
}

void validate(
	const sd::options& config)
{
	if (!(config.clip_level > 0.0 && config.clip_level <= 1.0)) {
		throw std::invalid_argument("clip level must be in (0, 1]");
	}
	if (config.derivative_floor < 0.0 || config.hard_derivative < 0.0 ||
		config.derivative_ratio <= 0.0 || config.baseline_window_ms <= 0.0 ||
		config.baseline_exclusion_ms < 0.0 || config.merge_gap_ms < 0.0 ||
		config.pre_roll_ms < 0.0 || config.post_roll_ms < 0.0 || config.max_repair_ms <= 0.0 ||
		config.lpc_order < 2 || config.lpc_context_ms <= 0.0 || config.lpc_mix < 0.0 ||
		config.lpc_mix > 1.0)
	{
		throw std::invalid_argument("invalid non-positive detector or reconstruction setting");
	}
}

void write_report(
	const std::filesystem::path& path, const lo::decoded_pcm& audio, const sd::Result& result)
{
	std::ofstream out(path);
	if (!out) {
		throw std::runtime_error("cannot open report: " + path.string());
	}
	out << "channel,kind,core_begin_frame,core_end_frame,repair_begin_frame,repair_end_frame,"
		   "start_seconds,end_seconds,duration_ms,score,original_peak,repaired_peak\n";
	out << std::setprecision(10);
	for (const auto& event : result.events) {
		const double start = static_cast<double>(event.repair_begin) / audio.sample_rate;
		const double end = static_cast<double>(event.repair_end + 1) / audio.sample_rate;
		out << event.channel << ',' << sd::to_string(event.kind) << ',' << event.core_begin << ','
			<< event.core_end << ',' << event.repair_begin << ',' << event.repair_end << ','
			<< start << ',' << end << ',' << (end - start) * 1000.0 << ',' << event.score << ','
			<< event.original_peak << ',' << event.repaired_peak << '\n';
	}
}

bool is_opus(
	const std::filesystem::path& path)
{
	std::string extension = path.extension().string();
	std::transform(extension.begin(), extension.end(), extension.begin(),
				   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return extension == ".opus";
}

bool is_wav(
	const std::filesystem::path& path)
{
	std::string extension = path.extension().string();
	std::transform(extension.begin(), extension.end(), extension.begin(),
				   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return extension == ".wav";
}

void append_frames(
	lo::decoded_pcm& destination, const lo::decoded_pcm& source, std::size_t begin, std::size_t end)
{
	if (begin > end || end > source.frames() || destination.sample_rate != source.sample_rate ||
		destination.channels != source.channels)
	{
		throw std::runtime_error("incompatible streamed audio block");
	}
	const std::size_t first = begin * source.channels;
	const std::size_t last = end * source.channels;
	destination.samples.insert(destination.samples.end(), source.samples.begin() + first,
							   source.samples.begin() + last);
}

lo::decoded_pcm take_frames(
	lo::decoded_pcm& source, std::size_t count)
{
	const std::size_t frames = std::min(count, source.frames());
	lo::decoded_pcm result = source;
	result.samples.clear();
	append_frames(result, source, 0, frames);
	source.samples.erase(source.samples.begin(), source.samples.begin() + frames * source.channels);
	return result;
}

template <typename reader_type, typename writer_type>
sd::Result process_streamed(
	const options& options, sd::options& config, lo::decoded_pcm& format,
	std::size_t& processed_frames)
{
	reader_type reader(options.input);
	format = reader.format();
	const std::size_t core_frames = std::max<std::size_t>(
		1, static_cast<std::size_t>(options.chunk_seconds * format.sample_rate));
	const std::size_t overlap_frames = std::max<std::size_t>(1, format.sample_rate / 2);
	config.worker_threads = options.threads;
	const std::size_t total_frames = reader.total_frames();
	lo::decoded_pcm history = format;
	lo::decoded_pcm future = format;
	std::optional<writer_type> writer;
	if (!options.dry_run) {
		writer.emplace(options.output, format);
	}

	sd::Result total;
	std::size_t output_frames = 0;
	std::size_t completed_chunks = 0;
	auto show_progress = [&]
	{
		const double seconds = static_cast<double>(output_frames) / format.sample_rate;
		std::cout << "\rprogress: " << std::fixed << std::setprecision(1) << seconds << " s";
		if (total_frames != 0) {
			std::cout << " / " << static_cast<double>(total_frames) / format.sample_rate << " s ("
					  << std::setprecision(0) << 100.0 * output_frames / total_frames << "%)";
		}
		std::cout << ", chunk " << completed_chunks << std::flush;
	};
	show_progress();
	while (true) {
		while (future.frames() < core_frames) {
			lo::decoded_pcm block = reader.read_frames(core_frames - future.frames());
			if (block.samples.empty())
				break;
			append_frames(future, block, 0, block.frames());
		}
		if (future.samples.empty())
			break;
		lo::decoded_pcm core = take_frames(future, core_frames);
		while (future.frames() < overlap_frames) {
			lo::decoded_pcm block = reader.read_frames(overlap_frames - future.frames());
			if (block.samples.empty())
				break;
			append_frames(future, block, 0, block.frames());
		}

		lo::decoded_pcm window = format;
		append_frames(window, history, 0, history.frames());
		append_frames(window, core, 0, core.frames());
		append_frames(window, future, 0, future.frames());
		const std::size_t left = history.frames();
		const std::size_t right = left + core.frames();
		sd::Result result = sd::process(window, config);
		for (const auto& event : result.events) {
			if (event.core_begin >= left && event.core_begin < right) {
				sd::event shifted = event;
				const std::size_t offset = output_frames - left;
				shifted.core_begin += offset;
				shifted.core_end += offset;
				shifted.repair_begin += offset;
				shifted.repair_end += offset;
				total.events.push_back(shifted);
			}
		}
		total.clipped_seed_frames += result.clipped_seed_frames;
		total.impulse_seed_frames += result.impulse_seed_frames;
		if (writer) {
			lo::decoded_pcm output = format;
			append_frames(output, window, left, right);
			writer->write_frames(output);
		}
		output_frames += core.frames();
		++completed_chunks;
		show_progress();
		history = std::move(core);
		if (history.frames() > overlap_frames) {
			history.samples.erase(history.samples.begin(),
								  history.samples.end() - overlap_frames * history.channels);
		}
	}
	std::cout << "\n";
	if (writer)
		writer->finish();
	processed_frames = output_frames;
	std::sort(total.events.begin(), total.events.end(),
			  [](const sd::event& a, const sd::event& b)
			  {
				  return a.repair_begin != b.repair_begin ? a.repair_begin < b.repair_begin
														  : a.channel < b.channel;
			  });
	return total;
}

} // namespace

int main(
	int argc, char** argv)
{
	try {
		const options options = parse_options(argc, argv);
		sd::options config = sd::preset_config(options.preset);
		apply_overrides(config, options);
		validate(config);
		if (!(options.chunk_seconds > 0.0)) {
			throw std::invalid_argument("chunk seconds must be positive");
		}

		lo::decoded_pcm audio;
		sd::Result result;
		std::size_t frame_count = 0;
		const bool streamed_opus = is_opus(options.input);
		const bool streamed_wav = is_wav(options.input);
		const bool streamed = streamed_opus || streamed_wav;
		if (streamed_opus) {
			result = process_streamed<lo::opus_reader, lo::opus_writer>(
				options, config, audio, frame_count);
		}
		else if (streamed_wav) {
			result = process_streamed<lo::wav_reader, lo::wav_writer>(
				options, config, audio, frame_count);
		}
		else {
			audio = lo::read_audio(options.input);
			result = sd::process(audio, config);
			frame_count = audio.frames();
		}
		std::cout << "input: " << options.input << '\n'
				  << "format: " << audio.sample_rate << " Hz, " << audio.channels << " channel(s), "
				  << audio.bits_per_sample << "-bit "
				  << (audio.encoding == lo::sample_encoding::pcm_integer ? "PCM" : "float") << '\n'
				  << "frames: " << frame_count << '\n'
				  << "preset: " << options.preset << '\n'
				  << "repair mode: " << sd::to_string(config.repair_mode) << '\n';
		if (streamed) {
			std::cout << (streamed_opus ? "Opus" : "WAV") << " mode: streamed ("
					  << options.chunk_seconds
					  << " s core, 500 ms overlap, "
					  << (options.threads == 0 ? "automatic" : std::to_string(options.threads))
					  << " worker(s))\n";
		}

		std::vector<std::size_t> per_channel(audio.channels, 0);
		for (const auto& event : result.events) {
			++per_channel[event.channel];
		}

		std::cout << "events: " << result.events.size() << '\n'
				  << "clipped seed frames: " << result.clipped_seed_frames << '\n'
				  << "impulse seed frames: " << result.impulse_seed_frames << '\n';
		for (std::uint16_t channel = 0; channel < audio.channels; ++channel) {
			std::cout << "channel " << channel << ": " << per_channel[channel] << " event(s)\n";
		}

		if (!options.report.empty()) {
			write_report(options.report, audio, result);
			std::cout << "report: " << options.report << '\n';
		}
		if (!options.dry_run && !streamed) {
			lo::write_audio(options.output, audio);
			std::cout << "output: " << options.output << '\n';
		}
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "error: " << error.what() << "\n\n";
		usage(std::cerr);
		return 1;
	}
}
