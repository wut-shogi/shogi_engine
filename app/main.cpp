#include <iostream>
#include "app.h"

int main(int argc, char* argv[]) {
  App app;
  app.Parse(argc, argv);
  return 0;
}
