require 'csv'

CSV.foreach("db/categories.csv", headers: true) do |row|
  puts row
  Category.find_or_create_by(slug: row['slug'], name: row['name'])
end
