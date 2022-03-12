require 'csv'

Recycler.all.each do |recycler|
  address, name, rest = recycler.description.split("\n")
  if name.nil?
    puts recycler.inspect
  else
    recycler.address = address
    recycler.name = name.gsub(/ для при[её]ма.*$/, '')
    recycler.save
  end
end
