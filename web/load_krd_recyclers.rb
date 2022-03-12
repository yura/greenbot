require 'csv'

cat_slugs =  %w[ plastik metal bymaga steklo odegda batareiki lampochki technika krishechki shini inoe zubnie_shetki gradusniki ]

krd = City.find_or_create_by(name: 'Краснодар')

CSV.foreach('db/krd_recyclers.csv', headers: true, col_sep: ';') do |row|
  puts row.inspect
  r = Recycler.find_or_create_by(description: row['name'], city: krd)
  r.lat, r.lon = *row['koordinat'].split(',')

  cat_slugs.each do |slug|
    if row[slug] == '1'
      puts "============ #{slug} #{row[slug]}: #{Category.find_by_slug(slug).inspect} "
      r.categories << Category.find_by_slug(slug)
    end
  end

  r.save
  puts r.errors.inspect
end
