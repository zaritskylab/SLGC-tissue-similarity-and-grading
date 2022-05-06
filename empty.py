# Closing parsers
for reader in parsers.values():
  if reader.m:
    reader.m.close()