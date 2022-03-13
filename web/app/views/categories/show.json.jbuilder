json.partial! "categories/category", category: @category
json.categories do
  json.array! @category.recyclers.limit(50), partial: "recyclers/recycler", as: :recycler
end

