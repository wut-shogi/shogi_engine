#include <functional>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>

class ArgsParser {
 public:
  ArgsParser(){};
  ArgsParser(const std::string& helpMessage);
  using ArgumentFunction = std::function<bool(const std::string&)>;
  using SubcommandParser = std::shared_ptr<ArgsParser>;

  void AddArgument(const std::string& argName,
                   ArgumentFunction func,
                   const std::string& helpMessage);

  void AddSubcommand(const std::string& subcommandName,
                     SubcommandParser subcommandParser);

  void SetMainFunction(std::function<void()> func);

  void Parse(int argc, char** argv, bool isSubcommand = false);

  void PrintHelp() const;

 private:
  std::unordered_map<std::string, ArgumentFunction> argsMap;
  std::function<void()> mainFunction;
  std::unordered_map<std::string, std::string> helpMap;
  std::unordered_map<std::string, SubcommandParser> subcommands;
  char nameValueSeparator = '=';
  std::string helpMessage;
};

class App {
 public:
  App();
  void Parse(int argc, char** argv);

 private:
  ArgsParser argsParser;
};