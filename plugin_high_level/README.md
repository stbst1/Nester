# Nester README

This is the README for the extension "Nester". Nester assists Python developers by automatically suggesting the top 5 type annotations whenever a type hint arrow (`->`) is encountered in the code. After a brief description, the README includes detailed sections on the features, requirements, settings, and more of the extension.

## Features

Nester offers real-time type inference suggestions within VS Code, activating when typing or navigating to a function's type hint arrow (`->`). Here are the specifics:

- **Type Hint Suggestions**: Displays the top 5 type annotations in a dropdown menu for easy selection.
- **Interactive Learning**: Includes a 'Learn More' button to generate a high-level program explanation, enhancing understanding of suggested type annotations.

## Requirements

Nester requires:
- Visual Studio Code 1.50.0 or higher.
- An active internet connection to fetch type annotations and generate explanations.

## Extension Settings

Nester contributes to the following settings via the `contributes.configuration` extension point:

* `nester.enable`: Enable/disable Nester.
* `nester.explainMode`: Set to `true` to automatically display explanations for selected annotations.

## Known Issues

- Delays in suggestions for large projects.
- Inaccuracies in type suggestions for highly dynamic or complex code.

## Release Notes

Inform users of updates to the extension.

### 1.0.0

Initial release:
- Real-time type annotation suggestions.
- Interactive explanations for type annotations.

### 1.0.1

Bug fixes:
- Improved suggestion accuracy.
- Reduced lag in displaying suggestions.

### 1.1.0

New features:
- Support for generating high-level program explanations directly in the editor.

---

## Working with Markdown

Visual Studio Code can be used to author your README. Here are some useful editor keyboard shortcuts:

* Split the editor (`Cmd+\` on macOS or `Ctrl+\` on Windows and Linux)
* Toggle preview (`Shift+Cmd+V` on macOS or `Shift+Ctrl+V` on Windows and Linux)
* Press `Ctrl+Space` (Windows, Linux, macOS) to see a list of Markdown snippets

## For more information

* [Visual Studio Code's Markdown Support](http://code.visualstudio.com/docs/languages/markdown)
* [Markdown Syntax Reference](https://help.github.com/articles/markdown-basics/)

**Enjoy!**
