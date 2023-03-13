#include <filesystem>
#include <string>

std::filesystem::path set_file_name(std::filesystem::path path, string new_name) {
    std::filesystem::path p(path.parent_path());
    p.append(new_name + path.extension().string());
    return p;
}

/**
 * @brief Appends a number at the end of a file if it already exists until a valid filename is found
 *
 * @param filename Desired filename
 *
 * @return Unique filename
 */
std::string get_unique_filename(std::string filename) {
    //? Maybe figure out a better way to do this

    std::filesystem::path p(filename);
    std::string name = p.stem();

    if (!std::filesystem::exists(p)) {
        return filename;
    } else {
        int suffix = 1;
        while (true) {
            std::filesystem::path p2 = set_file_name(p, name + "_" + to_string(suffix));

            if (!std::filesystem::exists(p2))
                break;

            suffix++;
        }
        return set_file_name(p, name + "_" + to_string(suffix)).string();
    }
}