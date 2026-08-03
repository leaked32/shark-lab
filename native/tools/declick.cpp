#include "shark/media.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

} // namespace

int main(
	int argc, char** argv)
{
	try {
		const options options = parse_options(argc, argv);
		sd::options config = sd::preset_config(options.preset);
		apply_overrides(config, options);
		validate(config);

		lo::decoded_pcm audio = lo::read_audio(options.input);
		std::cout << "input: " << options.input << '\n'
				  << "format: " << audio.sample_rate << " Hz, " << audio.channels << " channel(s), "
				  << audio.bits_per_sample << "-bit "
				  << (audio.encoding == lo::sample_encoding::pcm_integer ? "PCM" : "float") << '\n'
				  << "frames: " << audio.frames() << '\n'
				  << "preset: " << options.preset << '\n'
				  << "repair mode: " << sd::to_string(config.repair_mode) << '\n';

		const sd::Result result = sd::process(audio, config);
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
		if (!options.dry_run) {
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
