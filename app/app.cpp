#include "app.h"
#include "engine.h"

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

ArgsParser::ArgsParser(const std::string& helpMessage)
    : helpMessage(helpMessage) {}

void ArgsParser::AddArgument(const std::string& argName,
                             ArgumentFunction func,
                             const std::string& helpMessage) {
  argsMap[argName] = func;
  helpMap[argName] = helpMessage;
}

void ArgsParser::AddSubcommand(const std::string& subcommandName,
                               SubcommandParser subcommandParser) {
  subcommands[subcommandName] = subcommandParser;
}

void ArgsParser::SetMainFunction(std::function<void()> func) {
  mainFunction = func;
}

void ArgsParser::Parse(int argc, char** argv, bool isSubcommand) {
  if (!isSubcommand && argc == 1) {
    PrintHelp();
    return;
  }
  if (argc > 1 && subcommands.find(argv[1]) != subcommands.end()) {
    subcommands[argv[1]]->Parse(argc - 1, argv + 1, true);
  } else {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (i == 1 && arg == "help") {
        PrintHelp();
        return;
      }
      auto pos = arg.find(nameValueSeparator);
      std::string name, value;
      if (pos != std::string::npos) {
        name = arg.substr(0, pos);
        value = arg.substr(pos + 1);
      } else {
        name = arg;
        value = "";
      }

      if (argsMap.find(name) != argsMap.end()) {
        bool success = argsMap[name](value);
        if (!success) {
          std::cerr << "Error processing argument: " << name << std::endl;
        }
      } else {
        std::cerr << "Unknown argument: " << name << std::endl;
      }
    }
    mainFunction();
  }
}

void ArgsParser::PrintHelp() const {
  if (!subcommands.empty()) {
    std::cout << "Subcommands:" << std::endl;
  }
  for (const auto& kv : subcommands) {
    std::cout << kv.first << " - " << kv.second->helpMessage << std::endl;
  }
  std::cout << "Arguments:" << std::endl;
  std::cout << "help - display information about subcommands and arguments"
            << std::endl;
  for (const auto& kv : helpMap) {
    std::cout << kv.first << " - " << kv.second << std::endl;
  }
}

App::App() {
  argsParser = ArgsParser("Write help for more information");

  auto setDepth = [](const std::string& value) -> bool {
    try {
      int parsedValue = std::stoi(value);
      if (parsedValue >= 0 &&
          parsedValue <= std::numeric_limits<uint16_t>::max()) {
        uint16_t result = static_cast<uint16_t>(parsedValue);
        Engine::SetDepth(parsedValue);
      } else {
        return false;
      }
    } catch (...) {
      return false;
    }
    return true;
  };

  auto setTime = [](const std::string& value) -> bool {
    try {
      unsigned long parsedValue = std::stoul(value);
      if (parsedValue <= std::numeric_limits<uint32_t>::max()) {
        uint32_t result = static_cast<uint32_t>(parsedValue);
        Engine::SetTime(parsedValue);
      } else {
        return false;
      }
    } catch (...) {
      return false;
    }
    return true;
  };

  auto setPosition = [](const std::string& value) -> bool {
    Engine::SetPosition(value);
    return true;
  };

  auto setGPUcount = [](const std::string& value) -> bool {
    try {
      int parsedValue = std::stoi(value);
      Engine::SetGPUCount(parsedValue);
    } catch (...) {
      return false;
    }
    return true;
  };

  auto bestMoveGPUCommand =
      std::make_shared<ArgsParser>("Get best move using GPU");
  bestMoveGPUCommand->SetMainFunction([] { Engine::getBestMoveGPU(); });
  bestMoveGPUCommand->AddArgument("depth", setDepth,
                                  "Maximum depth used for calculations");
  bestMoveGPUCommand->AddArgument("time", setTime,
                                  "Maximum time for calculations");
  bestMoveGPUCommand->AddArgument("gpuCount", setGPUcount,
                                  "GPUs used for calculations");
  bestMoveGPUCommand->AddArgument("position", setPosition,
                                  "Starting position in SFEN string");
  argsParser.AddSubcommand("bestMoveGPU", bestMoveGPUCommand);

  auto bestMoveCPUCommand =
      std::make_shared<ArgsParser>("Get best move using CPU");
  bestMoveCPUCommand->SetMainFunction([] { Engine::getBestMoveCPU(); });
  bestMoveCPUCommand->AddArgument("depth", setDepth,
                                  "Maximum depth used for calculations");
  bestMoveCPUCommand->AddArgument("time", setTime,
                                  "Maximum time for calculations");
  bestMoveCPUCommand->AddArgument("position", setPosition,
                                  "Starting position in SFEN string");
  argsParser.AddSubcommand("bestMoveCPU", bestMoveCPUCommand);

  auto perftGPUCommand = std::make_shared<ArgsParser>("perft using GPU");
  perftGPUCommand->SetMainFunction([] { Engine::perftGPU(); });
  perftGPUCommand->AddArgument("depth", setDepth,
                               "Maximum depth used for calculations");
  perftGPUCommand->AddArgument("time", setTime,
                               "Maximum time for calculations");
  perftGPUCommand->AddArgument("gpuCount", setGPUcount,
                               "GPUs used for calculations");
  perftGPUCommand->AddArgument("position", setPosition,
                               "Starting position in SFEN string");
  argsParser.AddSubcommand("perftGPU", perftGPUCommand);

  auto perftCPUCommand = std::make_shared<ArgsParser>("perft using CPU");
  perftCPUCommand->SetMainFunction([] { Engine::perftCPU(); });
  perftCPUCommand->AddArgument("depth", setDepth,
                               "Maximum depth used for calculations");
  perftCPUCommand->AddArgument("time", setTime,
                               "Maximum time for calculations");
  perftCPUCommand->AddArgument("position", setPosition,
                               "Starting position in SFEN string");
  argsParser.AddSubcommand("perftCPU", perftCPUCommand);
}

void App::Parse(int argc, char** argv) {
  argsParser.Parse(argc, argv);
}
