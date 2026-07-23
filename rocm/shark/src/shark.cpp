
#include <boost/json/serialize.hpp>
#include <fstream>

#include "shark/shark.h"

void shark::forcely_print_vector(const std::vector<std::string> &input) {

	// std::cout << "\tdebug forcely_print_vector: ";
	for (size_t i = 0; i < input.size(); ++i) {
		// std::cout << i << ":";
		forcely_print_string(input.at(i));
		std::cout << ", ";
	}
	// std::cout << std::endl;
}

void shark::indented_print(std::string_view s, char c, int n) {}

std::string shark::file::tlm_node::print(unsigned short indent) const {
	std::ostringstream ss;
	std::string indent_str(indent, ' ');

	// --- children ---
	if (children) {
		for (const auto &[key, child] : *children) {
			ss << indent_str << "child:" << key << ":\n";
			ss << child.print(indent + 2);
		}
	}

	// --- doc (word wrapped, aligned) ---
	if (doc) {
		std::string_view text = *doc;

		std::string prefix = indent_str + "doc: ";
		std::string continuation(prefix.size(), ' ');

		bool first_line_global = true;

		size_t start = 0;
		while (start <= text.size()) {
			size_t end = text.find('\n', start);
			if (end == std::string_view::npos)
				end = text.size();

			std::string_view line = text.substr(start, end - start);

			// --- empty line (preserve it) ---
			if (line.empty()) {
				if (first_line_global) {
					ss << prefix << '\n';
					first_line_global = false;
				} else {
					ss << continuation << '\n';
				}
			} else {
				// --- word wrap THIS line only ---
				size_t pos = 0;
				bool first_line_local = true;

				while (pos < line.size()) {
					size_t line_start = pos;
					size_t len = 0;
					size_t last_space = std::string_view::npos;

					size_t limit = 128 - prefix.size();

					while (pos < line.size() && len < limit) {
						if (std::isspace(
						        static_cast<unsigned char>(line[pos]))) {
							last_space = pos;
						}
						++pos;
						++len;
					}

					if (pos < line.size() &&
					    last_space != std::string_view::npos &&
					    last_space > line_start) {
						pos = last_space + 1;
						len = last_space - line_start;
					}

					while (
					    pos < line.size() &&
					    std::isspace(static_cast<unsigned char>(line[pos]))) {
						++pos;
					}

					if (first_line_global) {
						ss << prefix;
						first_line_global = false;
					} else if (first_line_local) {
						ss << continuation;
					} else {
						ss << continuation;
					}

					ss << line.substr(line_start, len) << '\n';
					first_line_local = false;
				}
			}

			start = end + 1;
		}
	}

	return ss.str();
}

std::string shark::indent(std::string_view s, char c, int n) {
	if (s.empty() || n <= 0)
		return std::string(s);
	std::string prefix(n, c);
	std::string result;
	result.reserve(s.size() + prefix.size() * 32); // safe overestimate

	size_t pos = 0;
	while (pos < s.size()) {
		size_t nl = s.find('\n', pos);
		if (nl == std::string_view::npos) {
			result += prefix;
			result += s.substr(pos);
			break;
		}
		result += prefix;
		result += s.substr(pos, nl - pos + 1);
		pos = nl + 1;
	}
	return result;
}

void shark::forcely_print_string(const std::string &input) {

	// std::cout << "";
	for (size_t i = 0; i < input.size(); ++i) {
		std::cout << (char)(*(input.data() + i));
	}
	// std::cout << std::endl;
}

std::string shark::str_replace(std::string_view input, std::string_view from,
                              std::string_view to) {
	using T = decltype(input);

	const size_t len_from = from.size();
	const size_t len_to = to.size();
	const size_t len_input = input.size();

	size_t pre_count = 0;
	for (size_t occur = 0; occur = input.find(from, occur), occur != T::npos;) {
		occur += len_from;
		pre_count += 1ul;
	}

	if (pre_count == 0) {
		throw pinch("expression not fonud");
	}

	size_t output_capacity = len_input + (len_to - len_from) * pre_count;
	std::string output;
	output.resize(output_capacity);

	// std::println("{} {} {} {}", input, from, to, pre_count);

	size_t it_last = 0;
	size_t it_count = 0;
	for (size_t it = 0; it = input.find(from, it), it != T::npos;) {
		// std::println("{} ", output);

		// 1. Copy the buffer before the found substring to the output, the
		// begin of the buffer should be last time, last_it = it + len_from
		std::copy(input.begin() + it_last, input.begin() + it,
		          output.begin() + it_last + it_count * (len_to - len_from));
		// 2. Copy the new substring to the output
		std::copy(to.begin(), to.end(),
		          output.begin() + it + it_count * (len_to - len_from));

		it += len_from;
		it_last = it;

		it_count += 1;
	}
	// std::println("{} {}", output.size(), output.length(), it_last + it_count
	// * (len_to - len_from)); size() does not count the ending \0 by
	// std::string, but \0 I put is counted, so size() and length() can't be
	// used.
	// 3. Copy the tail buffer
	std::copy(input.begin() + it_last, input.end(),
	          output.begin() + it_last + it_count * (len_to - len_from));

	return output;
}

bool shark::is_empty_or_whitespace(const std::string &s) {
	return std::all_of(s.begin(), s.end(),
	                   [](unsigned char c) { return std::isspace(c); });
}

std::optional<std::string> shark::str_trim(const std::string &s) {

	if (std::all_of(s.begin(), s.end(),
	                [](unsigned char c) { return std::isspace(c); })) {
		return {};
	}

	auto start = std::find_if_not(
	    s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });

	auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) {
		           return std::isspace(c);
	           }).base();

	if (start >= end)
		return "";
	return std::string(start, end);
}

unsigned short shark::leading_space_count(const std::string &s) {
	unsigned short count = 0;
	auto start =
	    std::find_if_not(s.begin(), s.end(),
	                     [&count](unsigned char c) { return std::isspace(c); });

	return count;
}

void shark::remove_whitespace(std::string &s) {
	s.erase(std::remove_if(s.begin(), s.end(),
	                       [](unsigned char c) { return std::isspace(c); }),
	        s.end());
}

std::vector<std::string> shark::str_split(std::string_view input,
                                         std::string_view delimiter) {
	using T = decltype(input);

	const size_t len_delimiter = delimiter.size();
	size_t
	    last_occur; // if the delimiter is on begin (0), no allocation is needed
	size_t occur =
	    1; // if the delimiter is on begin (0), no allocation is needed
	size_t pre_count = 0;
	for (; occur = input.find(delimiter, occur), occur != T::npos;) {
		last_occur = occur;
		occur += len_delimiter + 1 /*the same reason*/;
		pre_count += 1ul;
	}
	if (last_occur + len_delimiter < input.size() - 1) {
		pre_count += 1;
	}

	if (pre_count == 0) {
		throw pinch("nothing available");
	}

	std::vector<std::string> output;
	output.reserve(pre_count);
	// log::info("reserved: {} {}", pre_count, last_occur);

	size_t it_last = 0;
	size_t it_count = 0;
	for (size_t it = 0; it = input.find(delimiter, it), it != T::npos;) {
		// 1. Copy the buffer before the found substring to the output, the
		// begin of the buffer should be last time, last_it = it + len_from
		if (it_last != it) {
			output.emplace_back(input.begin() + it_last, input.begin() + it);
		}

		it += len_delimiter;
		it_last = it;

		it_count += 1;
	}
	// 3. Copy the tail buffer
	if (it_last != input.size()) {
		output.emplace_back(input.begin() + it_last, input.end());
	}

	return output;
}

std::string shark::file::read(std::string_view path) {
	// Warning std::string_view is not null-terminated guaranteed.
	std::ifstream file(std::string(path), std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		shark::raise("Cannot open file: ", path);
	}

	std::string str((std::istreambuf_iterator<char>(file)),
	                std::istreambuf_iterator<char>());
	return str;
}

boost::json::object shark::file::read_json(std::string_view path) {
	namespace json = boost::json;
	auto raw = json::parse(read(path));
	return raw.as_object();
}

void shark::file::dump_json(std::string_view path,
                           const boost::json::object &js) {

	namespace json = boost::json;
	std::string pretty = json::serialize(js);

	std::ofstream(std::string(path)) << pretty;
}

std::vector<char> shark::file::read_binary(const std::string &filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Failed to open file: " + filename);
	size_t size = file.tellg();
	std::vector<char> buffer(size);
	file.seekg(0);
	file.read(buffer.data(), size);
	return buffer;
}

shark::file::tlm_node shark::file::tlm_node_read(std::string_view file_path) {
	const std::string identifier = "=^..^= ";
	std::string ctx = shark::file::read(file_path);

	decltype(ctx)::iterator ctx_begin = ctx.begin();

	using ty_indexes = std::vector<std::string>;
	ty_indexes indexes;

	tlm_node node;
	auto mk_doc = [](tlm_node &node, std::string_view doc,
	                 const ty_indexes &indexes) {
		tlm_node *idx = &node;
		for (const auto &index : indexes) {
			idx = &(*idx)[index];
		}
		idx->set_doc(doc);
	};

	auto decribe_id = [](std::string_view id, ty_indexes &indexes) {
		short inherit_from = -1;

		std::string_view::iterator id_begin = id.end();
		for (auto id_i = id.begin(); id_i != id.end(); ++id_i) {
			if (std::isspace(static_cast<unsigned char>(*id_i))) {
				continue;
			}
			if (inherit_from == -1) {
				if (*id_i == ':') {
					inherit_from = 1;
					continue;
				} else {
					inherit_from = 0;
					indexes.clear();
				}
			}
			if (id_begin == id.end()) {
				id_begin = id_i;
			}

			if (*id_i == ':') {
				indexes.emplace_back(id_begin, id_i);
				id_begin = id_i + 1;
			}
		}
	};

	for (size_t pos = 0; pos < ctx.size(); ++pos) {

		if (pos == 0 ? true : ctx.at(pos - 1) == '\n') {
			if (std::string_view(ctx.data() + pos, identifier.size())
			        .compare(identifier) == 0) {
				// line of the key
				if (!indexes.empty()) {
					auto tmp_begin = ctx_begin + 1;
					auto tmp_end = ctx.begin() + pos - 1;
					mk_doc(node,
					       tmp_end > tmp_begin
					           ? std::string_view{tmp_begin, tmp_end}
					           : "",
					       indexes);
				}

				size_t pos_next =
				    ctx.find_first_of('\n', pos); // Move to the next new line
				size_t ps_id_begin = pos + identifier.size();

				std::string_view id(ctx.data() + ps_id_begin,
				                    pos_next - ps_id_begin);
				decribe_id(id, indexes);

				// lit::log::info("id:{}", id);

				ctx_begin = ctx.begin() + pos_next;
			}
			// Not identifier line, don't care about it.'
		}
	}

	// node.print();
	return node;
}

float shark::math::random_float(const float begin, const float end) {
	auto &ins = math::instance();
	ins.dis.param(typename decltype(ins.dis)::param_type{begin, end});
	return ins.dis(ins.gen);
}

int shark::math::random_int(const int begin, const int end) {
	float r = shark::math::random_float();
	return begin + (int)(r * (end - begin));
}
