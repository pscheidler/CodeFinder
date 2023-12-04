# CodeFinder
Playing around with image rec for a puzzle in Frosthaven. Contains moderate Frosthaven spoilers!

`BWConvert.py` is used to convert the color image into stark black and white for contour matching

`MatchSymbols.py` is the main program for managing the symbols. It uses the system in `ContourContainer.py` to store 
the boxes containing the symbols, assign them to groups, and set them as active or not. The group state is intended
to be used to group together similar symbols. Active is just a property for display purposes.

TODO: Add in a Group to Letter dict, and display the letter of symbols in a group under the symbol in question

When the image is opened with `MatchSymbols.py`, the software will automatically run contour matching and get an
initial list of symbols. Identified symbols are boxed in Blue, Active symbols are in Green

All commands are a single letter. Note that for many commands, you enter the letter then select the symbols to act on:
* z - Run automatic grouping
*  l - Load saved settings
*  x - eXport settings
*  q - Quit
*  a - Add box for symbol (note, this command requires you click and drag the shape of the box)
*  d - Delete symbol (click in the symbol box)
*  s - Select, click on the symbol and select all symbols it is grouped with
*  g - Group, add selected symbols to the selected group
*  u - Ungroup, remove selected symbols from the active group, to the unclassified group
*  n - New group, select an ungrouped symbol and start a new group for it
*  y - Show all ungrouped symbols
*  m - Match, run the contour matching on the selected symbol. For debug only.
