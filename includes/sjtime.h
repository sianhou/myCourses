/**
  * Created by Sian Hou on 2024/9/7.
  * Copyright (c) 2024 RESEARCH INSTITUTE OF PETROLEUM EXPLORATION & DEVELOPMENT.
  * All rights reserved.
  *
  * Version 0.1: 2024/9/7
  */
#ifndef SJ_CORES_BASIC_SJTIME_H_
#define SJ_CORES_BASIC_SJTIME_H_

#include "map"
#include "string"
#include "vector"
#include "chrono"
#include "iostream"

namespace sj {

class MultiWatch {

  private:
    struct _OneWatch;

    using Ttype = std::chrono::system_clock::time_point;
    using Itype = std::map<std::string, _OneWatch>::iterator;

    struct _OneWatch {

        std::vector<std::pair<Ttype, Ttype>> _watch;

        void start() {
            auto new_time = std::pair<Ttype, Ttype>(std::chrono::system_clock::now(), std::chrono::system_clock::now());
            _watch.push_back(new_time);
            _num_start++;
        }

        void stop() {
            if (_num_stop + 1 == _num_start) {
                _watch[_watch.size() - 1].second = std::chrono::system_clock::now();
                _num_stop++;
            } else {
                std::cout << "Error in MultiWatch: record number dismatch!" << std::endl;
                exit(EXIT_FAILURE);
            }
        };

        float duration() {
            long sec = 0;
            if (_num_stop == _num_start) {
                for (auto it = _watch.begin(); it != _watch.end(); ++it) {
                    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(it->second - it->first);
                    sec += dur.count();
                }
            } else {
                std::cout << "Error in MultiWatch: record number dismatch!" << std::endl;
                exit(EXIT_FAILURE);
            }
            return sec / 1000000.0f;
        }

        size_t n_watch() {
            return _watch.size();
        }

        unsigned _num_start = 0, _num_stop = 0;
    };

  public:

    void reset() {
        for (auto &_watche : _watches) {
            _watche.second._watch.clear();
        }
    }

    void Insert(const std::string &elem) {
        _watches.insert(std::pair<std::string, _OneWatch>(elem, _OneWatch()));
    }

    _OneWatch &operator[](const std::string &elem) {
        auto it = _watches.find(elem);
        if (it != _watches.end()) {
            return _watches.find(elem)->second;
        } else {
            std::cout << "Error in MultiWatch: " << elem << " is not initialized!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

  private:

    std::map<std::string, _OneWatch> _watches;
};
}
#endif //SJ_CORES_BASIC_SJTIME_H_
